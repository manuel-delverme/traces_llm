import numpy as np
import pygame

from enjoy import HandwritingRecognizer


class BaseGUI:
    def __init__(self):
        self.token_motor_traces = [[], ]


    def move_mouse(self, x, y, t):
        char_motor_traces = self.token_motor_traces[-1]
        char_motor_traces.append((x, y, t))

    def new_char(self):
        self.token_motor_traces.append([])


class PygameGUI(BaseGUI):
    def __init__(self, model, user_interaction):
        import pygame
        pygame.init()
        # self.model = model
        self.user = user_interaction
        self.WIDTH, self.HEIGHT = 600, 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.recognizer = HandwritingRecognizer(model)
        super().__init__()

    def recognize_handwriting(self):
        token_motor_traces = np.array(self.token_motor_traces)
        prediction = self.recognizer.update_history_and_predict(token_motor_traces.copy())
        self.recognizer.next_token()
        self.token_motor_traces = [[], ]
        return prediction

    def run_once(self):
        should_continue = True
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                should_continue = False
            if event.type == pygame.MOUSEMOTION:
                pos = self.user.get_pos()
                if pos:
                    x, y = pos
                    t = pygame.time.get_ticks()
                    self.move_mouse(x, y, t)
                    pygame.draw.circle(self.surface, self.WHITE, (x, self.WIDTH - y), 10)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                prediction = self.recognize_handwriting()
                # TODO: process the prediction and display it
                self.user.reset(self.WIDTH, self.HEIGHT)
                print(prediction)

        self.window.fill(self.WHITE)
        self.window.blit(self.surface, (0, 0))
        pygame.display.flip()
        return should_continue

    def run(self):
        import pygame
        clock = pygame.time.Clock()
        self.user.reset(self.WIDTH, self.HEIGHT)
        while self.run_once():
            clock.tick(60)
        pygame.quit()
