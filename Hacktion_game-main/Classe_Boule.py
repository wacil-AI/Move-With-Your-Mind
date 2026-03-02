import pygame
from pathlib import Path

BLEU = (0, 100, 255)
IMAGES_DIR = Path(__file__).resolve().parent / "Images"

class Boule:
    def __init__(self, largeur, hauteur, scale):

        self.coeff_vitesse_clavier = 0.6
        self.coeff_vitesse_bci = 0.6
        
        # Position relative
        self.largeur = largeur
        self.hauteur = hauteur
        self.scale = scale

        self.x = largeur // 2
        self.y = int(hauteur - 100 * scale)

        self.rayon = int(25 * scale * 1.6)

        self.image_originale = pygame.image.load(str(IMAGES_DIR / "Cerveau.png")).convert_alpha()
        diam = self.rayon * 2
        self.image = pygame.transform.scale(self.image_originale, (diam, diam))

        # vitesses adaptées
        self.vitesse_clavier = 5 * scale * self.coeff_vitesse_clavier
        
        self.pas_bci = 5 * scale * self.coeff_vitesse_bci

    def redimensionner(self, largeur, hauteur, scale):
        """appelé quand la fenêtre change"""
        self.largeur = largeur
        self.hauteur = hauteur
        self.scale = scale

        self.y = int(hauteur - 100 * scale)
        self.rayon = int(25 * scale * 1.6)

        diam = self.rayon * 2
        self.image = pygame.transform.scale(self.image_originale, (diam, diam))

        self.vitesse_clavier = 5 * scale * self.coeff_vitesse_clavier
        self.pas_bci = 5 * scale * self.coeff_vitesse_bci

        # garde la boule dans l'écran
        self.x = max(self.rayon, min(largeur - self.rayon, self.x))

    def update_bci(self, commande):
        if commande == "gauche":
            self.x -= self.pas_bci
        elif commande == "droite":
            self.x += self.pas_bci

        self.x = max(self.rayon, min(self.largeur - self.rayon, self.x))

    def update_clavier(self, keys):
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.x -= self.vitesse_clavier
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.x += self.vitesse_clavier

        self.x = max(self.rayon, min(self.largeur - self.rayon, self.x))

    def draw(self, screen):
        rect = self.image.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(self.image, rect)
