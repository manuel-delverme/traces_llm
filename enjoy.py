import pygame
import numpy as np
import torch

from dataset import DataSpec
from text_only_baseline import GPT2FineTuning

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 600, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load the trained model
model = GPT2FineTuning(DataSpec(
    use_images=False,
    use_motor_traces=True,
))
# model.load_state_dict(torch.load("model.pt"))
model.eval()

window = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.Surface((WIDTH, HEIGHT))

# Initialize a list to store the mouse positions
mouse_positions = []

def main():
    clock = pygame.time.Clock()

    pygame.event.clear()
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                pygame.draw.circle(surface, WHITE, (x, y), 10)
                mouse_positions.append((x, y))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Convert the mouse positions to a PyTorch tensor and pass it through the model
                input_tensor = torch.FloatTensor(mouse_positions)
                prediction = model(input_tensor.unsqueeze(0))

                # TODO: process the prediction and display it
                # ...

        window.fill(WHITE)
        window.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
