import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pygame

from Menu import affichage_menu  # affichage_menu -> Fonction affichant le menu
from Jeu import affichage_jeu
from Victoire import affichage_victoire
from Calibrage import *


class BCIBridgeProcess:
    """Auto-start the LSL->prediction->UDP bridge when entering the game page."""

    def __init__(self, forced_model_path: Optional[str] = None, select_model: bool = False) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._project_root = Path(__file__).resolve().parent.parent
        self._bridge_script = self._project_root / "online_windowing.py"
        self._last_model_file = self._project_root / ".bci_last_model.txt"
        self._forced_model_path = Path(forced_model_path).expanduser().resolve() if forced_model_path else None
        self._select_model = bool(select_model)
        self._selection_done = False

    @staticmethod
    def _env(name: str, default: str) -> str:
        return os.environ.get(name, default).strip()

    def _enabled(self) -> bool:
        flag = self._env("BCI_AUTOSTART", "1").lower()
        return flag not in {"0", "false", "no", "off"}

    def set_model_path(self, model_path: Optional[str]) -> None:
        if not model_path:
            return
        p = Path(model_path).expanduser().resolve()
        if not p.exists():
            print(f"[BCI BRIDGE] Ignoring missing model path: {p}")
            return
        self._forced_model_path = p
        self._persist_last_model(p)
        print(f"[BCI BRIDGE] Active model set to: {p}")

    def _persist_last_model(self, model_path: Path) -> None:
        try:
            self._last_model_file.write_text(str(model_path), encoding="utf-8")
        except Exception as exc:
            print(f"[BCI BRIDGE] Warning: failed to persist last model path ({exc})")

    def _load_persisted_model(self) -> Optional[Path]:
        try:
            if not self._last_model_file.exists():
                return None
            text = self._last_model_file.read_text(encoding="utf-8").strip()
            if not text:
                return None
            p = Path(text).expanduser().resolve()
            if p.exists():
                return p
        except Exception:
            return None
        return None

    def _discover_model_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        candidates.extend(sorted(self._project_root.glob("*.pkl")))
        model_dir = self._project_root / "Hacktion_game-main" / "Model_simple"
        if model_dir.exists():
            candidates.extend(sorted(model_dir.glob("*.pkl")))

        dedup: dict[str, Path] = {}
        for p in candidates:
            if p.is_file():
                dedup[str(p.resolve())] = p.resolve()
        return sorted(dedup.values(), key=lambda p: p.stat().st_mtime, reverse=True)

    def _select_model_interactively(self) -> None:
        if self._selection_done or not self._select_model:
            return
        self._selection_done = True

        models = self._discover_model_candidates()
        if not models:
            print("[BCI BRIDGE] No model files found for selection.")
            return

        print("\n[BCI BRIDGE] Available model weights:")
        for idx, path in enumerate(models, start=1):
            rel = path.relative_to(self._project_root) if path.is_relative_to(self._project_root) else path
            print(f"  {idx}. {rel}")

        default = 1
        if not sys.stdin or not sys.stdin.isatty():
            self.set_model_path(str(models[default - 1]))
            print(f"[BCI BRIDGE] Non-interactive terminal, using default model: {self._forced_model_path}")
            return
        try:
            raw = input(f"[BCI BRIDGE] Select model index [default {default}]: ").strip()
        except EOFError:
            raw = ""

        choice = default
        if raw:
            try:
                choice = int(raw)
            except ValueError:
                choice = default
        if choice < 1 or choice > len(models):
            choice = default

        self.set_model_path(str(models[choice - 1]))
        print(f"[BCI BRIDGE] Selected model: {self._forced_model_path}")

    def _resolve_model_path(self) -> Optional[Path]:
        self._select_model_interactively()
        if self._forced_model_path is not None:
            return self._forced_model_path

        model_env = self._env("BCI_MODEL_PATH", "")
        if model_env:
            return Path(model_env).expanduser().resolve()

        persisted = self._load_persisted_model()
        if persisted is not None:
            return persisted

        default_model = self._project_root / "model_Data28_BIS.pkl"
        return default_model

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self._enabled():
            print("[BCI BRIDGE] Autostart disabled (BCI_AUTOSTART=0).")
            return
        if not self._bridge_script.exists():
            print(f"[BCI BRIDGE] Missing script: {self._bridge_script}")
            return

        stream_type = self._env("BCI_STREAM_TYPE", "EEG")
        stream_name = self._env("BCI_STREAM_NAME", "")
        source_id = self._env("BCI_SOURCE_ID", "")
        window_seconds = self._env("BCI_WINDOW_SECONDS", "1.0")
        stride_seconds = self._env("BCI_STRIDE_SECONDS", "0.1")
        overlap_seconds = self._env("BCI_OVERLAP_SECONDS", "")
        fixed_fs_hz = self._env("BCI_FIXED_FS_HZ", "")
        udp_host = self._env("BCI_UDP_HOST", "127.0.0.1")
        udp_port = self._env("BCI_UDP_PORT", "5005")
        health_interval = self._env("BCI_HEALTH_INTERVAL_SECONDS", "2.0")
        no_samples_warn = self._env("BCI_NO_SAMPLES_WARN_SECONDS", "3.0")
        min_conf = self._env("BCI_MIN_CONFIDENCE", "0.0")
        force_seven_channels = self._env("BCI_FORCE_7_CHANNELS", "1").lower() not in {"0", "false", "no", "off"}
        forced_channel_order = self._env("BCI_MODEL_CHANNEL_ORDER", "F3,F4,C3,Cz,C4,P3,P4")

        model_path = self._resolve_model_path()

        cmd = [
            sys.executable,
            str(self._bridge_script),
            "--stream-type",
            stream_type,
            "--window-seconds",
            window_seconds,
            "--udp-host",
            udp_host,
            "--udp-port",
            udp_port,
            "--min-confidence",
            min_conf,
            "--health-interval-seconds",
            health_interval,
            "--no-samples-warn-seconds",
            no_samples_warn,
        ]
        if stride_seconds:
            cmd.extend(["--stride-seconds", stride_seconds])
            if overlap_seconds:
                print(
                    "[BCI BRIDGE] Both BCI_STRIDE_SECONDS and BCI_OVERLAP_SECONDS are set; "
                    "using BCI_STRIDE_SECONDS."
                )
        elif overlap_seconds:
            cmd.extend(["--overlap-seconds", overlap_seconds])

        if stream_name:
            cmd.extend(["--stream-name", stream_name])
        if source_id:
            cmd.extend(["--source-id", source_id])
        if fixed_fs_hz:
            cmd.extend(["--fixed-fs-hz", fixed_fs_hz])
        if model_path is not None and model_path.exists():
            cmd.extend(["--model-path", str(model_path)])
        else:
            print(f"[BCI BRIDGE] Model file not found at {model_path}; bridge will run without prediction.")

        if force_seven_channels:
            cmd.extend(
                [
                    "--model-channel-order",
                    forced_channel_order,
                    "--strict-model-channel-order",
                ]
            )

        if self._env("BCI_USE_CAR", "0").lower() not in {"0", "false", "no", "off"}:
            cmd.append("--car")

        bp_low = self._env("BCI_BANDPASS_LOW_HZ", "")
        bp_high = self._env("BCI_BANDPASS_HIGH_HZ", "")
        if bp_low:
            cmd.extend(["--bandpass-low-hz", bp_low])
        if bp_high:
            cmd.extend(["--bandpass-high-hz", bp_high])

        if self._env("BCI_LIVE_PLOT", "0").lower() in {"1", "true", "yes", "on"}:
            cmd.append("--live-plot")

        extra_args = self._env("BCI_EXTRA_ARGS", "")
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        print(f"[BCI BRIDGE] Starting: {' '.join(cmd)}")
        self._proc = subprocess.Popen(cmd, cwd=str(self._project_root))

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return

        print("[BCI BRIDGE] Stopping.")
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        finally:
            self._proc = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run game and auto-start BCI bridge.")
    parser.add_argument("--model-path", type=str, default=None, help="Model weights path (.pkl) for online predictions")
    parser.add_argument("--select-model", action="store_true", help="Choose model weights interactively at startup")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pygame.init()

    info = pygame.display.Info()
    largeur = int(info.current_w * 0.9)
    hauteur = int(info.current_h * 0.9)

    screen = pygame.display.set_mode((largeur, hauteur), pygame.RESIZABLE)
    pygame.display.set_caption("Jeu BCI")
    clock = pygame.time.Clock()

    current_page = "menu"
    running = True
    temps_victoire = 0.0
    bridge = BCIBridgeProcess(forced_model_path=args.model_path, select_model=bool(args.select_model))

    try:
        while running:
            if current_page == "menu":
                result = affichage_menu(screen, clock, largeur, hauteur)
                if result == "go_to_jeu":
                    current_page = "jeu"
                elif result == "go_to_calibrage":
                    current_page = "calibrage"
                elif result == "quit":
                    running = False

            if current_page == "jeu":
                bridge.start()
                try:
                    result = affichage_jeu(screen, clock, largeur, hauteur)
                finally:
                    bridge.stop()

                if isinstance(result, tuple) and result[0] == "go_to_victoire":
                    temps_victoire = result[1]
                    current_page = "victoire"
                elif result == "go_to_menu":
                    current_page = "menu"
                elif result == "quit":
                    running = False

            if current_page == "calibrage":
                result = affichage_calibrage(screen, clock, largeur, hauteur)
                if result == "go_to_menu":
                    current_page = "menu"
                elif isinstance(result, tuple):
                    if len(result) > 0 and result[0] == "quit":
                        running = False
                    elif len(result) > 0 and result[0] == "go_to_menu":
                        payload = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
                        new_model = payload.get("model_path")
                        new_edf = payload.get("edf_path")
                        if new_edf:
                            print(f"[CALIB] Session EDF: {new_edf}")
                        if new_model:
                            bridge.set_model_path(str(new_model))
                        current_page = "menu"

            if current_page == "victoire":
                result = affichage_victoire(screen, clock, largeur, hauteur, temps_victoire)
                if result == "go_to_menu":
                    current_page = "menu"
                elif result == "quit":
                    running = False
    finally:
        bridge.stop()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

