import pygame
import socket
import time
from pygame.locals import *
from Outils_pygame import *
from Classe_Jeu import JeuBCI

class UdpBCIReceiver:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.setblocking(False)  # IMPORTANT : ne bloque jamais la boucle pygame
        self.last_value = None
        self.packet_count = 0
        self.command_count = 0
        self.last_packet_raw = None
        self.last_packet_monotonic = None
        self.last_command = None
        self.last_command_monotonic = None

    def poll(self):
        """Lit tous les paquets dispo (non bloquant) et renvoie la dernière commande."""
        cmd = None
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break  # plus rien à lire

            s = data.decode("utf-8").strip()
            now = time.monotonic()
            self.packet_count += 1
            self.last_packet_raw = s
            self.last_packet_monotonic = now

            if s == "-1":
                cmd = "gauche"
            elif s == "1":
                cmd = "droite"
            else:
                continue

            self.command_count += 1
            self.last_command = cmd
            self.last_command_monotonic = now
            self.last_value = cmd

        return cmd

    def health_snapshot(self):
        now = time.monotonic()
        packet_age_s = None
        if self.last_packet_monotonic is not None:
            packet_age_s = max(0.0, now - self.last_packet_monotonic)
        command_age_s = None
        if self.last_command_monotonic is not None:
            command_age_s = max(0.0, now - self.last_command_monotonic)
        return {
            "packet_count": int(self.packet_count),
            "command_count": int(self.command_count),
            "last_packet_raw": self.last_packet_raw,
            "last_command": self.last_command,
            "packet_age_s": packet_age_s,
            "command_age_s": command_age_s,
        }

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

def affichage_jeu(screen, clock, largeur, hauteur):

    running = True
    jeu = JeuBCI(largeur, hauteur)
    
    bci_udp = UdpBCIReceiver(ip="127.0.0.1", port=5005)
    status_font = pygame.font.Font(None, 28)
    allow_keyboard = True
    
    while running:

        for event in pygame.event.get():
            if event.type == QUIT:
                bci_udp.close()
                return "quit"

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    bci_udp.close()
                    return "go_to_menu"
                if event.key == K_k:
                    allow_keyboard = not allow_keyboard

            if event.type == VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), RESIZABLE)
                largeur, hauteur = event.w, event.h
                jeu.redimensionner(largeur, hauteur)

        commande_bci = bci_udp.poll()
        jeu.update(commande_bci, allow_keyboard=allow_keyboard)

        # ✅ victoire
        if jeu.temps_final is not None:
            bci_udp.close()
            return ("go_to_victoire", jeu.temps_final / 1000)

        screen.fill((200, 170, 255))
        jeu.draw(screen, largeur, hauteur)

        health = bci_udp.health_snapshot()
        packet_age_s = health["packet_age_s"]
        command_age_s = health["command_age_s"]
        if packet_age_s is None:
            udp_state = "WAITING"
        elif packet_age_s < 1.0:
            udp_state = "LIVE"
        else:
            udp_state = "STALE"
        last_cmd_txt = "none"
        if health["last_command"] is not None and command_age_s is not None:
            last_cmd_txt = f"{health['last_command']} ({command_age_s:.2f}s ago)"
        raw_txt = health["last_packet_raw"] if health["last_packet_raw"] is not None else "-"
        keyboard_txt = "ON" if allow_keyboard else "OFF"

        lines = [
            f"BCI UDP: {udp_state} | packets={health['packet_count']} valid_cmds={health['command_count']}",
            f"Last BCI cmd: {last_cmd_txt} | last_raw={raw_txt}",
            f"Keyboard fallback: {keyboard_txt} (press K to toggle)",
        ]

        panel_w = min(largeur - 20, 900)
        panel_h = 88
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((20, 20, 20, 170))
        screen.blit(panel, (10, 48))

        y = 54
        for line in lines:
            txt = status_font.render(line, True, (255, 255, 255))
            screen.blit(txt, (18, y))
            y += 26

        pygame.display.flip()
        clock.tick(60)
        
    bci_udp.close()
    return "go_to_menu"
