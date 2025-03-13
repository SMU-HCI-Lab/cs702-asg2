import copy
import random
import pygame
from dataclasses import dataclass


@dataclass
class PIDController:
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.1
    error_accumulator: float = 0
    prev_error: float = 0

    def calc_input(self, sp: float, pv: float, umin: float = -100, umax: float = 100) -> float:
        """Calculate the control signal.
        
        Args:
            sp: Set point
            pv: Process variable
            umin: Minimum control output
            umax: Maximum control output
            
        Returns:
            Control signal value
        """
        e = sp - pv
        P = self.Kp * e

        self.error_accumulator += e
        I = self.Ki * self.error_accumulator

        D = self.Kd * (e - self.prev_error)
        self.prev_error = e

        pid = P + I + D

        if pid < umin:
            u = umin
        elif pid > umax:
            u = umax
        else:
            u = pid

        # Uncomment for debugging:
        # print(f"P: {P:0.2f}, I: {I:0.2f}, D: {D:0.2f}, e: {e:0.2f}, u: {u:0.2f}")
        return u


@dataclass
class Bird:
    x: float  # x position
    y: float  # y position
    vx: float  # x velocity
    vy: float  # y velocity
    w: float = 20  # width
    h: float = 20  # height


def bird_motion(bird: Bird, u: float, dt: float, gravity: float = -50) -> Bird:
    """Updates the bird's position and velocity.
    
    Args:
        bird: Bird object to update
        u: Control input (upward force)
        dt: Time step
        gravity: Gravitational constant
        
    Returns:
        Updated Bird object
    """
    new_bird = copy.deepcopy(bird)
    new_bird.y = bird.y + bird.vy * dt
    new_bird.vy = bird.vy + (u + gravity) * dt
    return new_bird


@dataclass
class Pipe:
    x: float  # x position
    h: float  # height of bottom pipe
    w: float = 70  # width
    gap: float = 200  # gap between pipes


def pipe_motion(pipe: Pipe, vx: float, dt: float, screen_width: int = 400) -> tuple[Pipe, int]:
    """Updates the pipe position and generates new pipes when needed.
    
    Args:
        pipe: Pipe object to update
        vx: Horizontal velocity
        dt: Time step
        screen_width: Width of the screen
        
    Returns:
        Tuple of (updated Pipe object, score increment)
    """
    new_pipe = copy.deepcopy(pipe)
    new_pipe.x -= vx * dt

    d_score = 0
    if new_pipe.x < -pipe.w:
        new_pipe.x = screen_width
        new_pipe.h = random.randint(200, 300)
        d_score = 1
    return new_pipe, d_score


def calculate_control_signal(bird: Bird, pipe: Pipe, pid: PIDController, k: int) -> float:
    """Calculate the control signal for the bird.
    
    Args:
        bird: Current bird state
        pipe: Current pipe state
        pid: PID controller instance
        k: Current time step
        
    Returns:
        Control signal value
    """
    sp = pipe.h + pipe.gap / 2  # Set point is the middle of the gap
    pv = bird.y + bird.h / 2    # Process variable is the center of the bird
    u_jump = pid.calc_input(sp, pv)
    return u_jump


def main():
    # Pygame initialization and settings
    pygame.init()
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 600
    WHITE = (240, 240, 240)
    GREEN = (0, 200, 0)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird PID Controller")
    
    # Helper function to convert game coordinates to screen coordinates
    def transform(x, y):
        return (x, SCREEN_HEIGHT - y)

    # Initialize the PID controller
    pid = PIDController(0.8, 0.01, 0.1)

    # Bird initialization
    bird = Bird(50, 300, 30, 0)
    x, y = transform(bird.x, bird.y)
    bird_rect = pygame.Rect(x, y, bird.w, bird.h)

    # Pipe initialization
    pipe_height = random.randint(200, 300)
    pipe = Pipe(SCREEN_WIDTH - 50, pipe_height)

    # Clock and game state initialization
    clock = pygame.time.Clock()
    running = True
    fps = 30
    dt = 1 / fps
    score = 0
    k = 0  # Time step

    # Main game loop
    while running:
        screen.fill(WHITE)

        # Handle events
        u_jump = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    u_jump = 500

        # Calculate the control signal (AI control)
        u_jump = calculate_control_signal(bird, pipe, pid, k)

        # Update bird
        bird = bird_motion(bird, u_jump, dt)
        x, y = transform(bird.x, bird.y)
        bird_rect.x = x
        bird_rect.y = y

        # Update pipe
        pipe, d_score = pipe_motion(pipe, bird.vx, dt, SCREEN_WIDTH)
        score += d_score
        
        # Increase bird speed when scoring
        if d_score > 0:
            bird.vx += 10
        
        # Create pipe rectangles for rendering and collision
        x, h = pipe.x, pipe.h
        x_screen, y_screen = transform(x, h)
        bottom_pipe_rect = pygame.Rect(x_screen, y_screen, pipe.w, h)
        top_pipe_rect = pygame.Rect(x_screen, 0, pipe.w, SCREEN_HEIGHT - h - pipe.gap)

        # Draw game elements
        pygame.draw.rect(screen, GREEN, bird_rect)
        pygame.draw.rect(screen, GREEN, bottom_pipe_rect)
        pygame.draw.rect(screen, GREEN, top_pipe_rect)

        # Draw score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        # Collision detection
        if (bird_rect.colliderect(bottom_pipe_rect) or
                bird_rect.colliderect(top_pipe_rect) or
                bird.y + bird.h > 1.5 * SCREEN_HEIGHT or
                bird.y < -0.5 * SCREEN_HEIGHT):
            running = False

        # Update display
        pygame.display.update()
        clock.tick(fps)
        k += 1

    pygame.quit()


if __name__ == "__main__":
    main()