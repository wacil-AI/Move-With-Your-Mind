import pygame
import random
from pathlib import Path
from pygame.locals import *

from Classe_Obstacle import Obstacle
from Classe_Boule import Boule

# Couleurs RGB
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)
GRIS = (128, 128, 128)
LILAS = (245, 230, 255)
MAUVE = (230, 215, 250)
IMAGES_DIR = Path(__file__).resolve().parent / "Images"


def crop_to_alpha(surface: pygame.Surface, threshold: int = 1) -> pygame.Surface:
    """
    Recadre la surface sur la plus petite bounding box contenant des pixels alpha > threshold.
    IMPORTANT: pygame.surfarray renvoie un tableau alpha indexé (x, y) (largeur, hauteur).
    """
    surf = surface.convert_alpha()
    w, h = surf.get_width(), surf.get_height()

    alpha = pygame.surfarray.pixels_alpha(surf)  # shape (w, h) = (x, y)
    mask = alpha > threshold

    if not mask.any():
        del alpha
        return surf

    # ✅ Ordre CORRECT : (x, y)
    xs, ys = mask.nonzero()
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    del alpha

    # ✅ Clamp sécurité (au cas où)
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))

    rect = pygame.Rect(x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1))

    # Dernière sécurité : si rect sort, on intersecte avec l'image
    rect = rect.clip(pygame.Rect(0, 0, w, h))

    return surf.subsurface(rect).copy()


class JeuBCI:
    def __init__(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur

        self.coeff_vitesse_obstacles = 0.5

        self.base_largeur = 800
        self.base_hauteur = 600

        self.scale_x = largeur / self.base_largeur
        self.scale_y = hauteur / self.base_hauteur
        self.scale = min(self.scale_x, self.scale_y)

        # Route
        self.route_largeur = largeur // 3

        # -------- Ligne d'arrivée : chargement + recadrage + scale à la largeur de la route --------
        self.img_arrivee_orig = pygame.image.load(str(IMAGES_DIR / "Ligne_Arrivee.png")).convert_alpha()
        self.img_arrivee_trim = crop_to_alpha(self.img_arrivee_orig, threshold=1)

        self.img_arrivee = None
        self._rescale_arrivee_image()  # crée self.img_arrivee à la bonne largeur (route)
        # -----------------------------------------------------------------------------------------

        self.boule = Boule(largeur, hauteur, self.scale)
        self.obstacles = []

        self.vitesse_route = 4 * self.scale * self.coeff_vitesse_obstacles

        # Ligne d'arrivée (logique inchangée)
        self.arrivee_generee = False
        self.arrivee_rect = None
        self.arrivee_vitesse = self.vitesse_route

        self.temps_depart = pygame.time.get_ticks()
        self.bloque = False
        self.temps_final = None
        self.distance = 0
        self.ligne_arrivee = 8000

        self.score_font = pygame.font.Font(None, 36)
        self.debug_font = pygame.font.Font(None, 24)

        self.spawn_timer = 0
        self.derniere_commande = None
        self.compteur_commandes = 0

    def _rescale_arrivee_image(self):
        """Redimensionne la ligne d'arrivée pour qu'elle reste dans la largeur de la route (sans déformation)."""
        target_w = max(1, int(self.route_largeur))  # largeur exacte de la route
        ratio = target_w / self.img_arrivee_trim.get_width()
        new_w = int(self.img_arrivee_trim.get_width() * ratio)
        new_h = int(self.img_arrivee_trim.get_height() * ratio)

        self.img_arrivee = pygame.transform.scale(self.img_arrivee_trim, (new_w, new_h)).convert_alpha()
        self.img_arrivee.set_alpha(255)

    def redimensionner(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur

        self.scale_x = largeur / self.base_largeur
        self.scale_y = hauteur / self.base_hauteur
        self.scale = min(self.scale_x, self.scale_y)

        self.vitesse_route = 4 * self.scale * self.coeff_vitesse_obstacles
        self.arrivee_vitesse = self.vitesse_route

        # route
        self.route_largeur = largeur // 3
        route_x = (largeur - self.route_largeur) // 2

        # Rescale de l'image arrivée pour rester dans la route
        self._rescale_arrivee_image()

        # Si la ligne d'arrivée existe déjà, on recale sa position/largeur et sa hauteur (cohérente avec l'image)
        if self.arrivee_rect is not None:
            self.arrivee_rect.x = route_x
            self.arrivee_rect.width = self.route_largeur
            self.arrivee_rect.height = self.img_arrivee.get_height()

        self.score_font = pygame.font.Font(None, int(36 * self.scale))
        self.debug_font = pygame.font.Font(None, int(24 * self.scale))

        for obs in self.obstacles:
            obs.redimensionner(largeur, hauteur, route_x, self.route_largeur, self.scale)

        self.boule.redimensionner(largeur, hauteur, self.scale)

    # ✅ Méthode propre
    def nouvelle_commande_bci(self, commande):
        self.derniere_commande = commande
        self.compteur_commandes += 1
        self.boule.update_bci(commande)

    def update(self, commande_bci=None, allow_keyboard=True):
        keys = pygame.key.get_pressed()

        # ✅ Mouvement autorisé même si bloqué
        if commande_bci:
            self.nouvelle_commande_bci(commande_bci)
        elif allow_keyboard:
            self.boule.update_clavier(keys)

        # ✅ Limite la boule à la route
        route_x = (self.largeur - self.route_largeur) // 2
        limite_gauche = route_x + self.boule.rayon
        limite_droite = route_x + self.route_largeur - self.boule.rayon
        self.boule.x = max(limite_gauche, min(limite_droite, self.boule.x))

        # ✅ Spawn obstacles UNIQUEMENT si pas bloqué
        if not self.bloque:
            self.spawn_timer += 1
            if self.spawn_timer > random.randint(120, 300):
                route_x = (self.largeur - self.route_largeur) // 2
                self.obstacles.append(
                    Obstacle(
                        self.largeur,
                        self.hauteur,
                        route_x,
                        self.route_largeur,
                        self.scale
                    )
                )
                self.spawn_timer = 0

        # ✅ Update obstacles (figés si bloqué)
        vitesse_active = 0 if self.bloque else self.vitesse_route
        self.obstacles = [obs for obs in self.obstacles if not obs.update(vitesse_active)]

        # ✅ Déplacement ligne d'arrivée : elle descend même si bloqué
        if self.arrivee_generee and self.arrivee_rect and self.temps_final is None:
            self.arrivee_rect.y += self.arrivee_vitesse

        # ✅ Collision obstacles
        collision = False
        for obs in self.obstacles:
            if (abs(self.boule.x - (obs.x + obs.largeur/2)) < self.boule.rayon + obs.largeur/2 and
                abs(self.boule.y - (obs.y + obs.hauteur/2)) < self.boule.rayon + obs.hauteur/2):
                self.bloque = True
                collision = True
                break

        # ✅ Collision ligne arrivée = victoire
        if self.arrivee_rect and self.temps_final is None:
            if self.arrivee_rect.colliderect(
                pygame.Rect(
                    self.boule.x - self.boule.rayon,
                    self.boule.y - self.boule.rayon,
                    self.boule.rayon * 2,
                    self.boule.rayon * 2
                )
            ):
                self.temps_final = pygame.time.get_ticks() - self.temps_depart

        # ✅ Déblocage automatique
        if self.bloque and not collision:
            self.bloque = False

        # Distance progresse tant que pas encore généré la ligne
        if not self.bloque and not self.arrivee_generee:
            self.distance += vitesse_active

            if self.distance >= self.ligne_arrivee:
                self.arrivee_generee = True

                route_x = (self.largeur - self.route_largeur) // 2

                # Hitbox : largeur route, hauteur = hauteur réelle de l'image
                hauteur_arrivee = self.img_arrivee.get_height()
                self.arrivee_rect = pygame.Rect(
                    route_x,
                    -hauteur_arrivee,
                    self.route_largeur,
                    hauteur_arrivee
                )

    def draw(self, screen, largeur, hauteur):
        screen.fill(MAUVE)

        route_x = (largeur - self.route_largeur) // 2
        pygame.draw.rect(screen, LILAS, (route_x, 0, self.route_largeur, hauteur))

        # Dessiner ligne d'arrivée (dans la largeur de la route, pas déformée)
        if self.arrivee_rect:
            img_rect = self.img_arrivee.get_rect()
            img_rect.centerx = route_x + self.route_largeur // 2
            img_rect.top = self.arrivee_rect.top
            screen.blit(self.img_arrivee, img_rect)

        for obs in self.obstacles:
            obs.draw(screen)

        self.boule.draw(screen)

        temps = (pygame.time.get_ticks() - self.temps_depart) / 1000
        score_text = self.score_font.render(
            f"Temps: {temps:.1f}s | Distance: {self.distance:.0f}/{self.ligne_arrivee}",
            True, BLANC
        )
        screen.blit(score_text, (10, 10))
