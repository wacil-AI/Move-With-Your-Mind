import pygame
from pathlib import Path
pygame.init()

# Localisation de la police
BASE_DIR = Path(__file__).resolve().parent
font_path = str(BASE_DIR / "Polices" / "Pixel_Game_Font.otf")


def draw_text(text, font, color, surface, x, y):
    """
    Fonction qui permet rapidement d'écrire du texte sur une zone avec des paramètres précis.
    """
    textobj = font.render(text,1,color)
    textrect = textobj.get_rect()
    textrect.topleft = (x,y)
    surface.blit(textobj,textrect)


def draw_button(screen, rect, text, color_bg, color_text, font_name = font_path):
    """
    Fonction permettant de dessiner rapidement un bouton dynamique avec un texte intérieur qui s'adapte aux dimensions du bouton.
    """
    pygame.draw.rect(screen, color_bg, rect, border_radius=10)
    font = adapt_font(text, font_name, rect.width * 0.9, rect.height * 0.9)
    text_surface = font.render(text, True, color_text)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)


def adapt_font(text, font_name, w_max, h_max):
    """
    Fonction permettant d'adapter la taille de la police d'un texte en fonction de la taille de la surface sur laquelle il est écrit.
    """
    font_size = int(h_max)  # Commencer par la hauteur max possible
    while font_size > 3:
        if font_name is None:
            font = pygame.font.SysFont(None, font_size)  # police système
        else:
            font = pygame.font.Font(font_name, font_size)
        text_width, text_height = font.size(text)
        if text_width <= w_max and text_height <= h_max:
            return font
        font_size -= 1
    if font_name is None:
        return pygame.font.SysFont(None, 3)
    else:
        return pygame.font.Font(font_name, 3)

def blit_image_proportionnelle(screen, img_originale, rect):
    img_w, img_h = img_originale.get_size()
    rect_w, rect_h = rect.size

    ratio_img = img_w / img_h
    ratio_rect = rect_w / rect_h

    if ratio_img > ratio_rect:
        # Image plus large → on limite par largeur
        new_w = rect_w
        new_h = int(rect_w / ratio_img)
    else:
        # Image plus haute → on limite par hauteur
        new_h = rect_h
        new_w = int(rect_h * ratio_img)

    img_scaled = pygame.transform.smoothscale(img_originale, (new_w, new_h))

    # Centrer dans le rectangle
    img_rect = img_scaled.get_rect(center=rect.center)
    screen.blit(img_scaled, img_rect)


default_font = pygame.font.Font(font_path,20)
