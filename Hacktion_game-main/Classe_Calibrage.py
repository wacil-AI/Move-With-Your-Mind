import pygame
import random
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, local_clock

IMAGES_DIR = Path(__file__).resolve().parent / "Images"

class CalibrageBCI:
    def __init__(self, largeur, hauteur, marker_callback=None, trials_per_class: int = 30):
        self.largeur = largeur
        self.hauteur = hauteur
        self.marker_callback = marker_callback
        self.trials_per_class = max(1, int(trials_per_class))

        self.fleche_gauche = pygame.image.load(str(IMAGES_DIR / "Fleche_gauche.png")).convert_alpha()
        self.fleche_droite = pygame.image.load(str(IMAGES_DIR / "Fleche_droite.png")).convert_alpha()

        # optionnel : redimensionner
        self.fleche_gauche = pygame.transform.scale(self.fleche_gauche, (250,250))
        self.fleche_droite = pygame.transform.scale(self.fleche_droite, (250,250))

        # Boule
        self.boule_x = largeur // 2
        self.boule_y = hauteur - 120
        self.boule_rayon = 20

        # Obstacle
        self.obstacle = None
        self.obstacle_largeur = 80
        self.obstacle_hauteur = 40
        self.obstacle_y = 0

        # Séquence essais
        self.sequence = ["gauche"] * self.trials_per_class + ["droite"] * self.trials_per_class
        random.shuffle(self.sequence)
        self.index = 0

        # Timing
        self.phase = "pause"  # pause ou trial
        self.phase_start = pygame.time.get_ticks()

        # LSL
        info = StreamInfo(
            name="BCICalibration",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id="bci_calibration_001"
        )
        self.outlet = StreamOutlet(info)

    def _emit_marker(self, label: str) -> None:
        ts = local_clock()
        text = str(label)
        self.outlet.push_sample([text], ts)
        if callable(self.marker_callback):
            try:
                self.marker_callback(text, float(ts))
            except Exception:
                pass

    def start_trial(self):
        direction = self.sequence[self.index]
        self.phase = "trial"
        self.phase_start = pygame.time.get_ticks()

        # obstacle côté opposé
        if direction == "gauche":
            obstacle_x = self.largeur // 2 + 100
        else:
            obstacle_x = self.largeur // 2 - 180

        self.obstacle = pygame.Rect(
            obstacle_x, 0,
            self.obstacle_largeur, self.obstacle_hauteur
        )

        # LSL start
        self._emit_marker(f"start_{direction}")
        # Direction marker used for trial labels in EDF annotations.
        self._emit_marker(direction)
        print(direction)

    def end_trial(self):
        direction = self.sequence[self.index]

        # LSL end
        self._emit_marker(f"end_{direction}")
        self.index += 1
        self.phase = "pause"
        self.phase_start = pygame.time.get_ticks()

    def update(self):
        now = pygame.time.get_ticks()

        if self.index >= len(self.sequence):
            return "finish"

        if self.phase == "pause":
            if now - self.phase_start > 3000:
                self.start_trial()

        elif self.phase == "trial":
            elapsed = now - self.phase_start
            progress = elapsed / 3000

            direction = self.sequence[self.index]

            # mouvement boule
            if direction == "gauche":
                self.boule_x = self.largeur//2 - 150 * progress
            else:
                self.boule_x = self.largeur//2 + 150 * progress

            # descente obstacle
            self.obstacle.y = progress * (self.boule_y - 40)

            if elapsed > 3000:
                self.end_trial()

    def draw(self, screen):
        screen.fill((20,20,40))

        # route
        pygame.draw.rect(screen, (80,80,80),
                         (self.largeur//2 - 200, 0, 400, self.hauteur))

        # boule
        pygame.draw.circle(screen, (0,200,255),
                           (int(self.boule_x), self.boule_y),
                           self.boule_rayon)

        # obstacle
        if self.phase == "trial":
            pygame.draw.rect(screen, (255,80,80), self.obstacle)

            direction = self.sequence[self.index]
            font = pygame.font.Font(None, 120)

            if direction == "gauche":
                screen.blit(self.fleche_gauche, (150, self.hauteur//2-60))
            else:
                screen.blit(self.fleche_droite, (self.largeur-350, self.hauteur//2-60))
