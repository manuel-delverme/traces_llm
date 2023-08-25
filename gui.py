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
        self.event_counter = 0
        self.WIDTH, self.HEIGHT = 600, 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.recognizer = HandwritingRecognizer(model)
        super().__init__()

    def recognize_handwriting(self, original_traces=None):
        if not self.char_strokes[0]:
            return None, None

        token_motor_traces = np.array(self.char_strokes)
        # "num_chars x num_strokes=1 x num_points x (x, y, t)"
        # Right now with scripted user we have only one stroke per character
        # Eventually it will be a mess to figure out which stroke belongs to which character
        # but that's not today's problem
        token_motor_traces = np.expand_dims(token_motor_traces, axis=1)
        (num_chars, num_strokes, num_points, point_dims) = token_motor_traces.shape
        print(f"Got a state of {num_chars} characters, each of which, made of {num_strokes} strokes,"
              f" each stroke has {num_points} {point_dims}d points")

        traces = token_motor_traces.copy()

        if original_traces is not None:
            original_traces = self.validate_traces(original_traces, traces)

        print("Recognizing handwriting")
        predictions, p_tokens = self.recognizer.update_history_and_predict(traces, original_traces)
        print(f"Top 5 predictions:")
        for prediction, p_token in zip(predictions, p_tokens):
            # Tokenizer decode adds a strange G as space, replace it
            ascii_prediction = []
            for p in prediction:
                if p.isascii():
                    ascii_prediction.append(p)
                else:
                    ascii_prediction.append("_")
            ascii_prediction = "".join(ascii_prediction)
            print(f"{ascii_prediction}: {p_token}")

        # For the moment we reset the recognizer after each character
        # This loses the use of having a history, but it's a start
        self.recognizer.reset()
        self.char_strokes = []
        return prediction, p_token

    def validate_traces(self, original_traces, traces):
        # remove zero traces
        original_traces = original_traces[~np.all(original_traces == 0, axis=(1, 2))]
        original_shape = list(original_traces.shape)
        original_shape.insert(1, 1)
        original_shape[3] += 1  # add time
        assert tuple(original_shape) == traces.shape, (
            f"Original trace shape {original_traces.shape} is different from the one we got from the user"
            f" {traces.shape}"
        )
        traces_ = traces.squeeze(1).copy()
        traces_ = traces_[:, :, :2]  # remove time
        char_edge = np.min(traces_, axis=1)
        traces_ -= char_edge[:, np.newaxis, :]
        # Compare the traces
        for trace_idx in range(traces_.shape[0]):
            ui_trace = traces_[trace_idx]
            original_trace = original_traces[trace_idx]
            if np.any(ui_trace != original_trace):
                out = np.concatenate((ui_trace, original_trace, ui_trace - original_trace), axis=1)
                print(f"Trace {trace_idx} is different")
                print("UI trace, original trace, diff")
                print(out)
        return original_traces

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

                self.event_counter += 1
                # print("=" * 80)
                print("=" * 80)
                print(f"Event {self.event_counter}")
                self.print_state()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                original_trace = None
                if self._should_validate:
                    original_trace = self.user.postprocessed_token_trace

                prediction, p_token = self.recognize_handwriting(original_trace)
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
            # clock.tick(60)
            clock.tick(9999)
        pygame.quit()

    def validate_pipeline(self):
        self._should_validate = True
        self.run()
        self._should_validate = False

    def print_state(self):
        print("Stroke len:", [len(stroke) for stroke in self.char_strokes])
        print("Recognizer char_history:", list(
            t for t in self.recognizer.token_history if t != self.recognizer.tokenizer.pad_token_id
        ))
