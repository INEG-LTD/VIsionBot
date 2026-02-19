import threading
import time
import math
from rich.live import Live
from rich.text import Text
from rich.console import Console

class TextAnimator:
    def __init__(self, text, effect="rainbow", speed=0.05):
        self.text = text
        self.effect = effect
        self.speed = speed
        self.running = False
        self.thread = None
        self.console = Console()

    # -------------------------
    # COLOR UTILITIES
    # -------------------------

    def _rainbow_color(self, t):
        r = int((math.sin(t) + 1) * 127.5)
        g = int((math.sin(t + 2) + 1) * 127.5)
        b = int((math.sin(t + 4) + 1) * 127.5)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _fade_color(self, t, start=(255,0,0), end=(0,0,255)):
        fade = (math.sin(t) + 1) / 2
        r = int(start[0] + (end[0] - start[0]) * fade)
        g = int(start[1] + (end[1] - start[1]) * fade)
        b = int(start[2] + (end[2] - start[2]) * fade)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _pulse_color(self, t):
        brightness = (math.sin(t) + 1) / 2
        value = int(255 * brightness)
        return f"#{value:02x}{value:02x}{value:02x}"

    def _gradient_text(self, t):
        styled = Text()
        for i, char in enumerate(self.text):
            offset = t + i * 0.3
            color = self._rainbow_color(offset)
            styled.append(char, style=color)
        return styled

    # -------------------------
    # RENDER LOOP
    # -------------------------

    def _render(self):
        with Live(refresh_per_second=120, console=self.console, transient=True) as live:
            t = 0
            while self.running:
                if self.effect == "rainbow":
                    color = self._rainbow_color(t)
                    styled = Text(self.text, style=f"bold {color}")

                elif self.effect == "fade":
                    color = self._fade_color(t)
                    styled = Text(self.text, style=f"bold {color}")

                elif self.effect == "pulse":
                    color = self._pulse_color(t)
                    styled = Text(self.text, style=f"bold {color}")

                elif self.effect == "gradient":
                    styled = self._gradient_text(t)

                else:
                    styled = Text(self.text)

                live.update(styled)
                t += 0.1
                time.sleep(self.speed)


    # -------------------------
    # CONTROL METHODS
    # -------------------------

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._render, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


    def hide(self):
        self.console.clear()
