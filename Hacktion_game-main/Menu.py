import pygame
from pathlib import Path

from pygame.locals import *  # On importe les commandes pygame qui relient le code aux pilotes de l'ordi (clavier, souris, etc.)

from Outils_pygame import *  # On importe les fonctions précodées par moi-même utiles au développement d'une interface pygame

from sys import exit

IMAGES_DIR = Path(__file__).resolve().parent / "Images"


def scale_proportionnel(img_orig, cible_w=None, cible_h=None):
    """Retourne une surface redimensionnée en gardant les proportions."""
    w, h = img_orig.get_size()
    ratio = w / h

    if cible_w is None and cible_h is None:
        return img_orig

    if cible_w is None:
        new_h = cible_h
        new_w = int(new_h * ratio)
    elif cible_h is None:
        new_w = cible_w
        new_h = int(new_w / ratio)
    else:
        # si on donne les deux, on prend la contrainte la plus forte
        ratio_cible = cible_w / cible_h
        if ratio > ratio_cible:
            new_w = cible_w
            new_h = int(new_w / ratio)
        else:
            new_h = cible_h
            new_w = int(new_h * ratio)

    return pygame.transform.smoothscale(img_orig, (new_w, new_h))


def affichage_menu(screen, clock, largeur, hauteur):
    """
    Fonction affichant un menu avec un titre : "Menu"
    et un bouton central : "Génération du labyrinthe".
    """
    running = True

    # --- IMAGES (chargées une seule fois) ---
    img_jeu_orig = pygame.image.load(str(IMAGES_DIR / "Jeu.png")).convert_alpha()
    img_calib_orig = pygame.image.load(str(IMAGES_DIR / "Calibrage.png")).convert_alpha()
    img_logo_orig = pygame.image.load(str(IMAGES_DIR / "Logo.png")).convert_alpha()

    while running :
        # ✅ 1️⃣ CALCULS BOUTONS TOUJOURS EN PREMIER (avant events)
        bouton_largeur = largeur // 3
        bouton_hauteur = hauteur // 8
        espacement = largeur // 20
        x_jeu = (largeur - 2*bouton_largeur - espacement) // 2
        x_calib = x_jeu + bouton_largeur + espacement
        y_boutons = hauteur * 2 // 5
        
        # Créer Rects pour draw_button ET collision
        bouton_jeu_rect = pygame.Rect(x_jeu, y_boutons, bouton_largeur, bouton_hauteur)
        bouton_calib_rect = pygame.Rect(x_calib, y_boutons, bouton_largeur, bouton_hauteur)

        # Nettoyage de l'écran principal
        screen.fill((200, 170, 255))

        logo_h = int(0.7 * hauteur) # logo
        img_logo = scale_proportionnel(img_logo_orig, cible_h=logo_h)

        logo_rect = img_logo.get_rect(midtop=(largeur // 2, 20))
        screen.blit(img_logo, logo_rect)

        # --- TAILLE DES BOUTONS (plus grands) ---
        bouton_w = int(0.35 * largeur)   # 35% largeur écran (plus petit qu’avant)

        img_jeu = scale_proportionnel(img_jeu_orig, cible_w=bouton_w)
        img_calib = scale_proportionnel(img_calib_orig, cible_w=bouton_w)

        # --- POSITION GAUCHE / DROITE ---
        jeu_rect = img_jeu.get_rect(center=(int(largeur * 0.30), int(hauteur * 0.55)))
        calib_rect = img_calib.get_rect(center=(int(largeur * 0.70), int(hauteur * 0.55)))

        # --- AFFICHAGE ---
        screen.blit(img_jeu, jeu_rect)
        screen.blit(img_calib, calib_rect)

        # On récupère les coordonnées du curseur de la souris
        mx, my = pygame.mouse.get_pos()       
        click = False

        for event in pygame.event.get():
            # Si croix cochée on quitte l'interface
            if event.type == QUIT:
                return "quit"

            # Si bouton escape touché on quitte l'interface
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return "quit"

            # Adapte la fenêtre au redimensionnement
            if event.type == VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), RESIZABLE)
                largeur, hauteur = event.w, event.h
                
            # ✅ 2️⃣ CLIC TESTÉ DIRECT DANS LA BOUCLE (Rects définis !)
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if jeu_rect.collidepoint(mx, my):
                        return "go_to_jeu"
                    elif calib_rect.collidepoint(mx, my):
                        return "go_to_calibrage"
                    click = True  # Garde pour compatibilité

        
        pygame.display.flip()
        # 60 FPS
        clock.tick(60)

