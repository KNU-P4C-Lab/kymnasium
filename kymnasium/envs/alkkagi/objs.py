import pygame
import pymunk
import math
from typing import Tuple
from .consts import PLAYER_BLACK
from kymnasium.common.color import Color


class Stone:
    RADIUS = 15
    MASS = 10
    ELASTICITY = 0.85

    def __init__(self, x: float, y: float, player: int, index: int):
        moment = pymunk.moment_for_circle(Stone.MASS, 0, Stone.RADIUS)

        self._body = pymunk.Body(Stone.MASS, moment)
        self._body.position = (x, y)

        self._shape = pymunk.Circle(self._body, Stone.RADIUS)
        self._shape.elasticity = Stone.ELASTICITY

        self._player = player
        self._index = index

        self.active = True

    def draw(self, surface: pygame.Surface):
        if not self.active:
            return

        font = pygame.font.SysFont(None, 24)
        if self._player == PLAYER_BLACK:
            pygame.draw.circle(surface, Color.BLACK, self._body.position, Stone.RADIUS)  # Black
            text_surface = font.render(str(self._index), True, Color.WHITE)
        else:
            pygame.draw.circle(surface, Color.WHITE, self._body.position, Stone.RADIUS)  # White
            pygame.draw.circle(surface, Color.BLACK, self._body.position, Stone.RADIUS, 1)  # Black outline
            text_surface = font.render(str(self._index), True, Color.BLACK)

        text_rect = text_surface.get_rect(center=self.position_)
        surface.blit(text_surface, text_rect)

    def apply_impulse(self, impulse: Tuple[float, float] = (0, 0)):
        self._body.apply_impulse_at_local_point(impulse, (0, 0))

    def stop(self):
        self._body.velocity = (0, 0)

    @property
    def position_(self):
        return self._body.position

    @property
    def index_(self):
        return self._index

    @property
    def player_(self):
        return self._player

    @property
    def shape_(self):
        return self._shape

    @property
    def body_(self):
        return self._body

    @property
    def velocity_(self):
        return self._body.velocity


class Obstacle:
    ELASTICITY = 0.25

    def __init__(self, x: float, y: float, width: float, height: float):
        self._width = width
        self._height = height

        self._body = pymunk.Body(body_type=pymunk.Body.STATIC)  # Static body (doesn't move)
        self._body.position = (x, y)

        self._shape = pymunk.Poly.create_box(self._body, (width, height))
        self._shape.elasticity = Obstacle.ELASTICITY  # Less bouncy for wooden board feel

    def draw(self, surface: pygame.Surface):
        x, y = self._body.position
        rect = pygame.Rect(x - self._width / 2, y - self._height / 2, self._width, self._height)
        pygame.draw.rect(surface, Color.BRASS, rect)
        pygame.draw.rect(surface, Color.BLACK, rect, 1)  # Black outline

    @property
    def size_(self):
        return self._width, self._height

    @property
    def position_(self):
        return self._body.position

    @property
    def shape_(self):
        return self._shape

    @property
    def body_(self):
        return self._body


class SlingShot:
    POWER_LINE_LENGTH = 100

    def __init__(self,
                 pos_start: Tuple[float, float] | pymunk.Vec2d | None = None,
                 pos_end: Tuple[float, float] | pymunk.Vec2d | None = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.wait = False

    def draw(self, surface: pygame.Surface):
        if self.pos_start is None or self.pos_end is None:
            return

        sx, sy = self.pos_start
        ex, ey = self.pos_end
        dx, dy = sx - ex, sy - ey
        distance = math.hypot(dx, dy)

        if distance <= 0:
            return

        nx, ny = dx / distance, dy / distance

        len_arrow, len_dash = 75, 5
        n_dashes = int(len_arrow / (2 * len_dash))

        ex, ey = sx + nx * len_arrow, sy + ny * len_arrow

        for i in range(n_dashes):
            start_ratio = i * (2 * len_dash) / len_arrow
            end_ratio = min(start_ratio + len_dash / len_arrow, 1)
            dash_start = (
                sx + (ex - sx) * start_ratio,
                sy + (ey - sy) * start_ratio
            )
            dash_end = (
                sx + (ex - sx) * end_ratio,
                sy + (ey - sy) * end_ratio
            )
            pygame.draw.line(surface, Color.LIGHT_GREY, dash_start, dash_end, 2)

        arrow_size = 10
        arrow_angle = 30  # degrees

        # Calculate arrow head points
        angle = pygame.math.Vector2(0, 0).angle_to(pygame.math.Vector2(nx, ny))
        l_angle, r_angle = math.radians(angle + arrow_angle), math.radians(angle - arrow_angle)

        arrow_left = (
            ex - arrow_size * math.cos(l_angle),
            ey - arrow_size * math.sin(l_angle)
        )
        arrow_right = (
            ex - arrow_size * math.cos(r_angle),
            ey - arrow_size * math.sin(r_angle)
        )

        pygame.draw.line(surface, Color.LIGHT_GREY, (ex, ey), arrow_left, 2)
        pygame.draw.line(surface, Color.LIGHT_GREY, (ex, ey), arrow_right, 2)

        len_power = min(distance, SlingShot.POWER_LINE_LENGTH)
        power_end = (
            sx - nx * len_power,
            sy - ny * len_power
        )

        pygame.draw.line(surface, Color.LIGHT_GREY, (sx, sy), power_end, 3)
