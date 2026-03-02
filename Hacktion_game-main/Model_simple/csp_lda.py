"""
Model_simple/csp_lda.py
================================================================================
Modele CSP + LDA — Interface simple

    model = CSPLDA()
    model.fit(X, y)          # entraine
    model.predict(X)         # predit les labels
    model.score(X, y)        # accuracy
    model.cv_evaluate(X, y)  # 10-fold CV

Donnees attendues :
    X : (n_trials, 7, n_times)   signal EEG — 7 electrodes
    y : (n_trials,)              labels 'left_hand' / 'right_hand'

Fenetre : 3s  |  Bande : 8-30 Hz  |  CV : 10-fold

Usage :
    python Model_simple/csp_lda.py                     # 0009 + 0010
    python Model_simple/csp_lda.py --edf 0010.edf      # session unique
    python Model_simple/csp_lda.py --edf 0009.edf 0010.edf
"""

import sys
import os
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone
from mne.decoding import CSP
import mne

# ─── Chemins ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PARENT_DIR)

#from load_edf_trials import CHANNEL_MAP, LABEL_MAP, SFREQ_TARGET
#from train_fbcsp_edf import reject_artifacts

# Mapping annotation → label chaîne de caractères
LABEL_MAP = {
    "Annotation/droite": "right_hand",   # droite = positif
    "Annotation/gauche": "left_hand",    # gauche = négatif
    "droite": "right_hand",
    "gauche": "left_hand",
    "start_droite": "right_hand",
    "start_gauche": "left_hand",
    "Annotation/start_droite": "right_hand",
    "Annotation/start_gauche": "left_hand",
    # "Rest" est ignoré
}

# ─── Mapping canaux EDF → noms d'électrodes ──────────────────────────────────
# ← Adapter si le câblage de votre casque est différent
CHANNEL_MAP = {
    "Ch1":  "F3",
    "Ch2":  "F4",
    "Ch3":  "C3",
    "Ch4":  "Cz",
    "Ch5":  "C4",
    "Ch6":  "P3",
    "Ch7":  "P4",
    # Ch8–Ch11 = canaux hardware (accel, batterie…) → ignorés
}

# =============================================================================
#  Rejection artefacts
# =============================================================================

def reject_artifacts(X, y, z_thresh=3.5):
    ptp   = (X.max(axis=-1) - X.min(axis=-1)).max(axis=-1)
    mu, s = ptp.mean(), ptp.std()
    seuil = mu + z_thresh * s
    keep  = ptp < seuil
    n_rm  = (~keep).sum()
    if n_rm:
        print(f"  Rejet artefacts : {n_rm}/{len(X)} trials supprimes (seuil={seuil:.2e})")
    else:
        print(f"  Rejet artefacts : aucun (seuil={seuil:.2e})")
    return X[keep], y[keep]

# ─── Parametres ───────────────────────────────────────────────────────────────
FMIN      = 8.0    # Hz — bande optimale 0010 (gridsearch)
FMAX      = 30.0   # Hz
TMIN      = 0.5    # s apres le trigger
TMAX      = 3.5    # s — fenetre de 3s
N_COMP    = 4      # composantes CSP (optimal gridsearch)
N_FOLDS   = 10     # CV


# =============================================================================
#  Runtime config persistence (train -> online inference)
# =============================================================================

def attach_training_config(model, ch_names, sfreq, trial_ptp_stats=None, cv_stats=None):
    """Attach training-time preprocessing/runtime constraints to saved model."""
    cfg = {
        "window_seconds": float(TMAX - TMIN),
        "tmin_seconds": float(TMIN),
        "tmax_seconds": float(TMAX),
        "bandpass_low_hz": float(FMIN),
        "bandpass_high_hz": float(FMAX),
        "fixed_fs_hz": float(sfreq),
        "apply_car": False,
        "channel_names": [str(ch) for ch in ch_names],
        "channel_order": [str(ch) for ch in ch_names],
        "label_space": ["left_hand", "right_hand"],
    }
    if isinstance(trial_ptp_stats, dict):
        cfg["train_trial_ptp_stats"] = dict(trial_ptp_stats)
    if isinstance(cv_stats, dict):
        cfg["cv_stats"] = dict(cv_stats)
    model.training_config_ = cfg
    # Redundant explicit attributes for backward/robust loading.
    model.expected_channel_names_ = list(cfg["channel_order"])
    model.training_window_seconds_ = float(cfg["window_seconds"])
    model.training_fixed_fs_hz_ = float(cfg["fixed_fs_hz"])
    model.training_bandpass_hz_ = (float(cfg["bandpass_low_hz"]), float(cfg["bandpass_high_hz"]))
    model.training_apply_car_ = bool(cfg["apply_car"])
    if isinstance(trial_ptp_stats, dict):
        model.training_trial_ptp_stats_ = dict(trial_ptp_stats)
    if isinstance(cv_stats, dict):
        model.training_cv_stats_ = dict(cv_stats)
    return model


# =============================================================================
#  Chargement EDF — fenetre 3s
# =============================================================================

def load_edf(edf_path: str, apply_artifact_reject: bool = True):
    """
    Charge un fichier EDF et retourne X, y prets pour model.fit().

    Applique :
      - Bandpass 8-30 Hz (mu + beta)
      - Segmentation en trials de 3s apres chaque annotation
      - Rejet d artefacts (optionnel, z-score > 3.5 sur le peak-to-peak)

    Retourne
    --------
    X : (n_trials, 7, n_times)   float64
    y : (n_trials,)              str  'left_hand' / 'right_hand'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Renommage des canaux
    rename = {old: new for old, new in CHANNEL_MAP.items()
              if old in raw.ch_names}
    if rename:
        raw.rename_channels(rename)

    eeg_names = [v for v in CHANNEL_MAP.values() if v in raw.ch_names]
    if not eeg_names:
        raise RuntimeError(f"Aucun canal EEG trouve dans {edf_path}")

    raw.pick(eeg_names)
    raw.set_channel_types({ch: "eeg" for ch in eeg_names})

    # Bandpass 8-30 Hz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.filter(FMIN, FMAX, method="fir", fir_design="firwin", verbose=False)

    """
        # Resampling si necessaire
        if raw.info["sfreq"] != SFREQ_TARGET:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.resample(SFREQ_TARGET, verbose=False)
    """
    # Segmentation en epochs de 3s
    available_ann = set(str(a) for a in raw.annotations.description)
    matched_ann = [ann for ann in LABEL_MAP.keys() if ann in available_ann]
    if not matched_ann:
        raise RuntimeError(
            f"Aucune annotation compatible trouvee dans {edf_path}. "
            f"Disponibles={sorted(available_ann)}"
        )
    event_id = {ann: i + 1 for i, ann in enumerate(matched_ann)}
    label_lookup = {ann: LABEL_MAP[ann] for ann in matched_ann}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, _ = mne.events_from_annotations(
            raw, event_id=event_id, verbose=False
        )

    if len(events) == 0:
        raise RuntimeError(f"Aucune annotation trouvee dans {edf_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=TMIN, tmax=TMAX,
            baseline=None, preload=True, verbose=False,
        )

    X        = epochs.get_data()
    inv      = {v: k for k, v in event_id.items()}
    y        = np.array([label_lookup[inv[e]] for e in epochs.events[:, 2]])
    ch_names = epochs.ch_names

    # Rejet artefacts (optionnel)
    if apply_artifact_reject:
        X, y = reject_artifacts(X, y, z_thresh=3.5)
    else:
        print(f"  Rejet artefacts : desactive ({len(X)} trials gardes)")

    return X, y, ch_names, raw.info["sfreq"]


# =============================================================================
#  Modele CSP + LDA
# =============================================================================

class CSPLDA:
    """
    Modele CSP + LDA pour la classification Gauche / Droite.

    Interface :
        model.fit(X, y)          -> entraine sur X, y
        model.predict(X)         -> labels predits
        model.score(X, y)        -> accuracy (float)
        model.cv_evaluate(X, y)  -> resultats 10-fold CV (dict)

    X : (n_trials, n_channels, n_times)
    y : (n_trials,)  — 'left_hand' / 'right_hand'  OU  0 / 1
    """

    def __init__(self, n_components: int = N_COMP, n_folds: int = N_FOLDS):
        self.n_components = n_components
        self.n_folds      = n_folds
        self.le_          = LabelEncoder()
        self._pipe        = None

    def _make_pipe(self) -> Pipeline:
        return Pipeline([
            ("csp", CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ])

    def _encode(self, y: np.ndarray) -> np.ndarray:
        if y.dtype.kind in ("U", "S", "O"):           # labels string
            return self.le_.fit_transform(y)
        return np.asarray(y, dtype=int)

    # ── API publique ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLDA":
        """
        Entraine le modele CSP+LDA sur (X, y).

        X : (n_trials, n_channels, n_times)
        y : labels string ou entiers
        """
        Y = self._encode(y)
        self._pipe = self._make_pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipe.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les labels predits (string si fit() avec strings)."""
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_pred = self._pipe.predict(X)
        if len(self.le_.classes_) > 0:
            return self.le_.inverse_transform(Y_pred)
        return Y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne P(gauche), P(droite) pour chaque trial."""
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._pipe.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy sur (X, y)."""
        self._check_fitted()
        Y = self.le_.transform(y) if y.dtype.kind in ("U","S","O") else y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self._pipe.score(X, Y))

    def cv_evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        10-fold StratifiedKFold CV.

        Retourne un dict avec :
          mean_acc   : accuracy moyenne
          std_acc    : ecart-type
          fold_accs  : accuracy par fold
          y_pred_cv  : predictions agregees (pour matrice de confusion)
          y_true_cv  : vrais labels agregees
        """
        Y = self._encode(y)
        n_splits = min(self.n_folds, len(Y) // 2)
        if n_splits < self.n_folds:
            print(f"  [!] Seulement {len(Y)} trials -> CV {n_splits}-fold")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_accs  = []
        y_pred_all = []
        y_true_all = []

        for train_idx, test_idx in cv.split(X, Y):
            pipe = self._make_pipe()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X[train_idx], Y[train_idx])
                y_pred = pipe.predict(X[test_idx])

            acc = float((y_pred == Y[test_idx]).mean())
            fold_accs.append(acc)
            y_pred_all.extend(y_pred.tolist())
            y_true_all.extend(Y[test_idx].tolist())

        return {
            "mean_acc":  float(np.mean(fold_accs)),
            "std_acc":   float(np.std(fold_accs)),
            "fold_accs": fold_accs,
            "n_folds":   n_splits,
            "y_pred_cv": np.array(y_pred_all),
            "y_true_cv": np.array(y_true_all),
            "classes":   self.le_.classes_,
        }

    def save(self, path: str):
        """Sauvegarde le modele entraine sur disque."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Modele sauvegarde -> {path}")

    @staticmethod
    def load(path: str) -> "CSPLDA":
        """Charge un modele depuis le disque."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"  Modele charge <- {path}")
        return model

    def _check_fitted(self):
        if self._pipe is None:
            raise RuntimeError("Appelle model.fit(X, y) avant de predire.")


# =============================================================================
#  Rapport et figure
# =============================================================================

def print_report(label: str, cv_result: dict):
    acc = cv_result["mean_acc"]
    std = cv_result["std_acc"]
    n   = cv_result["n_folds"]
    print(f"\n  {'='*50}")
    print(f"  {label}")
    print(f"  {'='*50}")
    print(f"  CV {n}-fold")
    print(f"  Accuracy : {acc*100:.1f}%  +/-  {std*100:.1f}%")
    print(f"  Par fold : {[f'{v*100:.0f}%' for v in cv_result['fold_accs']]}")

    classes = cv_result["classes"]
    cm = confusion_matrix(cv_result["y_true_cv"], cv_result["y_pred_cv"])
    print(f"  Matrice de confusion :")
    for i, row in enumerate(cm):
        lbl = classes[i] if len(classes) > i else str(i)
        print(f"    {lbl:12s} : {row}")


def plot_results(results: dict, out_path: str):
    """
    Figure 2 panneaux :
      - Gauche : accuracy par session (barres + erreur)
      - Droite : matrice de confusion agregee
    """
    labels   = list(results.keys())
    accs     = [results[l]["mean_acc"] * 100 for l in labels]
    stds     = [results[l]["std_acc"]  * 100 for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Barres accuracy
    ax = axes[0]
    colors = ["#1565C0", "#1976D2", "#E65100", "#E65100", "#2E7D32"][:len(labels)]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85,
                  edgecolor="black", lw=0.7,
                  yerr=stds, capsize=6,
                  error_kw={"lw": 1.5, "capthick": 1.5})
    ax.axhline(50, color="gray", ls="--", lw=1.2, label="Chance 50%")
    ax.set_ylim(20, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CSP + LDA — 10-fold CV\n"
                 f"Fenetre {int(TMAX)}s  |  Bande {int(FMIN)}-{int(FMAX)} Hz",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.0f}%", ha="center", fontsize=10, fontweight="bold")

    # Matrice de confusion agregee (toutes sessions)
    ax2 = axes[1]
    y_true_all = np.concatenate([r["y_true_cv"] for r in results.values()])
    y_pred_all = np.concatenate([r["y_pred_cv"] for r in results.values()])
    classes    = results[labels[0]]["classes"]
    cm         = confusion_matrix(y_true_all, y_pred_all)
    disp       = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[c.replace("_hand", "") for c in classes]
    )
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Matrice de confusion\n(toutes sessions agregees)",
                  fontweight="bold")

    fig.suptitle("Model_simple — CSP + LDA",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure -> {out_path}")


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSP+LDA simple — Gauche vs Droite"
    )
    parser.add_argument(
        "--edf", nargs="+", default=["0009.edf", "0010.edf"],
        help="Un ou plusieurs fichiers EDF"
    )
    args = parser.parse_args()

    # Resolution chemins
    edfs = []
    for f in args.edf:
        for cand in [f, os.path.join(_PARENT_DIR, f)]:
            if os.path.exists(cand):
                edfs.append(os.path.abspath(cand))
                break
        else:
            print(f"[ERREUR] Fichier introuvable : {f}")
            sys.exit(1)

    print("=" * 55)
    print("  Model_simple — CSP + LDA")
    print(f"  Fenetre  : {TMAX}s  |  Bande : {FMIN}-{FMAX} Hz")
    print(f"  N comp   : {N_COMP}  |  CV : {N_FOLDS}-fold")
    print(f"  Capteurs : F3 F4 C3 Cz C4 P3 P4 (7 electrodes)")
    print("=" * 55)

    all_results = {}

    for edf_path in edfs:
        name = os.path.basename(edf_path).replace(".edf", "")
        print(f"\n  Chargement {name}.edf ...")
        # 1) Full events for final model fit (deployment model).
        X_all, y_all, ch_names, sfreq = load_edf(edf_path, apply_artifact_reject=False)
        print(
            f"  X all events : {X_all.shape}   "
            f"({(y_all=='left_hand').sum()} G / {(y_all=='right_hand').sum()} D)"
        )

        # 2) Optional artifact-rejected subset for CV reporting.
        X_cv, y_cv = reject_artifacts(X_all, y_all, z_thresh=3.5)
        print(
            f"  X CV subset : {X_cv.shape}   "
            f"({(y_cv=='left_hand').sum()} G / {(y_cv=='right_hand').sum()} D)"
        )
        print(f"  Canaux : {ch_names}")
        trial_ptp = np.max(np.ptp(X_all, axis=2), axis=1)
        trial_ptp_stats = {
            "median": float(np.median(trial_ptp)),
            "p95": float(np.percentile(trial_ptp, 95.0)),
            "max": float(np.max(trial_ptp)),
            "min": float(np.min(trial_ptp)),
        }
        print(
            "  PTP train stats: "
            f"median={trial_ptp_stats['median']:.3e} "
            f"p95={trial_ptp_stats['p95']:.3e} "
            f"max={trial_ptp_stats['max']:.3e}"
        )

        model = CSPLDA(n_components=N_COMP, n_folds=N_FOLDS)
        unique_cv = np.unique(y_cv)
        if X_cv.shape[0] < 4 or unique_cv.size < 2:
            print(
                "  [!] CV subset invalide apres rejet artefacts -> "
                "fallback CV sur all events."
            )
            X_cv_eval, y_cv_eval = X_all, y_all
        else:
            X_cv_eval, y_cv_eval = X_cv, y_cv

        cv_result = model.cv_evaluate(X_cv_eval, y_cv_eval)
        all_results[name] = cv_result
        print_report(name, cv_result)

        # Entraine sur TOUTES les donnees (all events) et sauvegarde
        model.fit(X_all, y_all)
        cv_stats = {
            "mean_acc": float(cv_result.get("mean_acc", np.nan)),
            "std_acc": float(cv_result.get("std_acc", np.nan)),
            "n_folds": int(cv_result.get("n_folds", 0)),
        }
        attach_training_config(
            model,
            ch_names,
            sfreq,
            trial_ptp_stats=trial_ptp_stats,
            cv_stats=cv_stats,
        )
        model.training_total_events_ = int(X_all.shape[0])
        model.training_cv_events_ = int(X_cv_eval.shape[0])
        model.save(os.path.join(_SCRIPT_DIR, f"model_{name}.pkl"))

    # Fusion si plusieurs fichiers
    if len(edfs) > 1:
        print(f"\n  Chargement fusion ...")
        Xs, ys = [], []
        for edf_path in edfs:
            X, y, _, sf = load_edf(edf_path)
            Xs.append(X)
            ys.append(y)
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        print(f"  X fusion : {X_all.shape}")

        model_fusion = CSPLDA(n_components=N_COMP, n_folds=N_FOLDS)
        cv_result    = model_fusion.cv_evaluate(X_all, y_all)
        all_results["Fusion"] = cv_result
        print_report("Fusion", cv_result)

        # Exemple : entrainer sur 0010, predire sur 0009
        print(f"\n  Exemple model.fit() / model.predict() :")
        X9,  y9,  ch9,  sf9 = load_edf(edfs[0])
        X10, y10, _, _ = load_edf(edfs[1])
        model_cross = CSPLDA()
        model_cross.fit(X10, y10)
        attach_training_config(model_cross, ch9, sf9)
        y_pred = model_cross.predict(X9)
        acc    = model_cross.score(X9, y9)
        print(f"  model.fit(X_0010, y_0010)")
        print(f"  model.predict(X_0009) -> {y_pred[:5]} ...")
        print(f"  model.score(X_0009, y_0009) -> {acc*100:.1f}%")

    # Figure
    out_fig = os.path.join(_SCRIPT_DIR, "results.png")
    plot_results(all_results, out_fig)

    print(f"\n{'='*55}")
    print(f"  Termine.")
    print(f"{'='*55}")
