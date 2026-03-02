import pygame
from pygame.locals import *

def affichage_victoire(screen, clock, largeur, hauteur, temps):

    running = True
    font_titre = pygame.font.Font(None, 80)
    font_score = pygame.font.Font(None, 50)

    while running:

        for event in pygame.event.get():
            if event.type == QUIT:
                return "quit"

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return "go_to_menu"

        screen.fill((30, 180, 80))

        titre = font_titre.render("VICTOIRE !", True, (255,255,255))
        score = font_score.render(f"Temps : {temps:.2f} secondes", True, (255,255,255))

        screen.blit(titre, (largeur//2 - titre.get_width()//2, hauteur//3))
        screen.blit(score, (largeur//2 - score.get_width()//2, hauteur//2))

        pygame.display.flip()
        clock.tick(60)
