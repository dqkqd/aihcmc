import sys
import pygame
from PIL import Image

pygame.init()


if __name__ == "__main__":
    arg = sys.argv[1]
    img = Image.open(arg)

    screen = pygame.display.set_mode(img.size)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                print(f"[{x}, {y}], ")

        image = pygame.image.load(arg)
        screen.blit(image, (0, 0))
        pygame.display.update()



