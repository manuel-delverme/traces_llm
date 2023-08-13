import numpy as np
import pygame

from preprocess import HandwritingRecognizer


class BaseGUI:
    def __init__(self):
        self.char_strokes = []

    def track_move_mouse(self, x, y, t):
        stroke_motor_traces = self.char_strokes[-1]
        stroke_motor_traces.append((x, y, t))

    def new_stroke(self):
        self.char_strokes.append([])


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
        if not self.char_strokes[0]:
            return None, None

        token_motor_traces = np.array(self.char_strokes)
        # "num_chars x num_strokes=1 x num_points x (x, y, t)"
        # Right now with scripted user we have only one stroke per character
        # Eventually it will be a mess to figure out which stroke belongs to which character
        # but that's not today's problem
        token_motor_traces = np.expand_dims(token_motor_traces, axis=1)

        traces = token_motor_traces.copy()

        prediction, p_token = self.recognizer.update_history_and_predict(traces)
        self.recognizer.next_token()
        # For the moment we reset the recognizer after each character
        # This loses the use of having a history, but it's a start
        self.recognizer.reset()
        self.char_strokes = []
        return prediction, p_token

    def process_event_queue(self):
        should_continue = True
        events = [self.user.get_event(), ]

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.new_stroke()
            if event.type == pygame.QUIT:
                should_continue = False
                break
            elif event.type == pygame.MOUSEMOTION:
                # Replace the event with the one from the user object
                x, y = event.pos
                t = pygame.time.get_ticks()
                self.track_move_mouse(x, y, t)
                pygame.draw.circle(self.surface, self.WHITE, (x, self.WIDTH - y,), 10)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                prediction, p_token = self.recognize_handwriting()
                # TODO: process the prediction and display it
                self.user.reset(self.WIDTH, self.HEIGHT)
                # Clear the screen
                self.surface.fill(self.BLACK)
                print(prediction, p_token)

        self.window.fill(self.WHITE)
        self.window.blit(self.surface, (0, 0))
        pygame.display.flip()
        return should_continue

    def run(self):
        import pygame
        clock = pygame.time.Clock()
        self.user.reset(self.WIDTH, self.HEIGHT)
        while self.process_event_queue():
            clock.tick(60)
        pygame.quit()
