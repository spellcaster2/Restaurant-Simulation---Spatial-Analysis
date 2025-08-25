import simpy
import pygame
import random
import numpy as np
import sys
import heapq
import math
import logging
from datetime import datetime
import csv
from collections import deque

# =========================
# --- Pygame & Visual -----
# =========================
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
SIDEBAR_WIDTH = 250
CANVAS_WIDTH = SCREEN_WIDTH - SIDEBAR_WIDTH
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (240, 240, 240)
RED = (200, 0, 0)
GREEN = (0, 180, 0)
BLUE = (0, 0, 200)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREY = (225, 225, 225)
AXIS_GREY = (190, 190, 190)

# =========================
# --- Globals -------------
# =========================
LAYOUT = {}
NUM_WAITERS = 1
WAITER_SPEED = 1.2       # m/s
CUSTOMER_SPEED = WAITER_SPEED * 1.5
SIM_SPEED_FACTOR = 60
AVG_WAITER_SERVING_TIME = 2.0  # minutes
AVG_DWELL_TIME = 30.0          # minutes

RESTAURANT_WIDTH = 20.0
RESTAURANT_HEIGHT = 15.0

seated_customer_wait_times = []
customers_served_count = 0
all_customers = []
all_waiters = []
ASSETS = {}
SCALE_FACTOR = (CANVAS_WIDTH - 100) / RESTAURANT_WIDTH

# grid/obstacle map
OBSTACLE_MAP = None  # numpy 2D array [gw][gh] of 0/1
OBSTACLE_RES = 1     # grid cells per meter (>=1). Keep 1 for speed; Theta* gives any-angle paths.

# UI toggles
SHOW_GRID = True
SHOW_HELP = True

# --- Demo validator memory (press D) ---
DEMO_RESULTS = []   # list of dict rows for UI display

# =========================
# --- Logging -------------
# =========================
LOG_TO_CONSOLE = True
LOG_FILE = "sim_log.csv"
KPI_LOG_FILE = "kpi_log.csv"   # <--- separate KPI log file
EVENT_LOG = deque(maxlen=200000)

def init_logger():
    if LOG_TO_CONSOLE:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    # main event log header
    with open(LOG_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_time", "sim_min", "actor", "event", "details"])
    # KPI log header
    with open(KPI_LOG_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "wall_time", "sim_min",
            "served", "active", "backlog",
            "avg_seated_wait_min", "avg_backlog_wait_min"
        ])

def log_event(env, actor, event, details=""):
    wall = datetime.now().isoformat(timespec="seconds")
    simt = getattr(env, "now", 0.0)
    row = [wall, f"{simt:.3f}", actor, event, details]
    EVENT_LOG.append(row)
    if LOG_TO_CONSOLE:
        logging.info(f"[t={simt:8.3f} min] [{actor:<12}] {event}: {details}")

def flush_log_to_file():
    if not EVENT_LOG:
        return
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerows(EVENT_LOG)
    EVENT_LOG.clear()

def log_kpis(env):
    backlog_customers = [c for c in all_customers if c.status == 'queuing']
    active_customers = [c for c in all_customers if c.status in ['seated', 'eating']]
    avg_seated_wait = np.mean(seated_customer_wait_times) if seated_customer_wait_times else 0.0
    current_backlog_waits = [env.now - c.arrival_time for c in backlog_customers] if env else []
    avg_backlog_wait = np.mean(current_backlog_waits) if current_backlog_waits else 0.0

    # human-readable snapshot to main event log
    log_event(
        env, "KPI", "snapshot",
        f"served={customers_served_count} active={len(active_customers)} "
        f"backlog={len(backlog_customers)} avg_seated_wait={avg_seated_wait:.2f} "
        f"avg_backlog_wait={avg_backlog_wait:.2f}"
    )

    # structured row to KPI CSV
    with open(KPI_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{getattr(env, 'now', 0.0):.3f}",
            customers_served_count,
            len(active_customers),
            len(backlog_customers),
            round(float(avg_seated_wait), 3),
            round(float(avg_backlog_wait), 3),
        ])

# =========================
# --- Utility -------------
# =========================
def reset_simulation_state():
    """Resets all global variables for a new simulation."""
    global LAYOUT, NUM_WAITERS, WAITER_SPEED, CUSTOMER_SPEED, SIM_SPEED_FACTOR, seated_customer_wait_times
    global AVG_WAITER_SERVING_TIME, AVG_DWELL_TIME, all_customers, all_waiters, customers_served_count
    global RESTAURANT_WIDTH, RESTAURANT_HEIGHT, OBSTACLE_MAP, DEMO_RESULTS

    LAYOUT = {"kitchen": None, "reception": None, "entry": None, "tables": {}, "arrival_rates": {}}
    NUM_WAITERS, WAITER_SPEED, SIM_SPEED_FACTOR = 1, 1.2, 60
    CUSTOMER_SPEED = WAITER_SPEED * 1.5
    AVG_WAITER_SERVING_TIME, AVG_DWELL_TIME = 2.0, 30.0
    RESTAURANT_WIDTH, RESTAURANT_HEIGHT = 20.0, 15.0
    seated_customer_wait_times = []
    customers_served_count = 0
    all_customers, all_waiters = [], []
    OBSTACLE_MAP = None
    DEMO_RESULTS = []

def update_scale_factor():
    """Calculates the scale factor to fit the restaurant dimensions within the canvas."""
    global SCALE_FACTOR
    if RESTAURANT_WIDTH > 0 and RESTAURANT_HEIGHT > 0:
        scale_x = (CANVAS_WIDTH - 100) / RESTAURANT_WIDTH
        scale_y = (SCREEN_HEIGHT - 100) / RESTAURANT_HEIGHT
        SCALE_FACTOR = min(scale_x, scale_y)
    else:
        SCALE_FACTOR = 1

def scale_pos(coords):
    return (int(coords[0] * SCALE_FACTOR) + 50, int(coords[1] * SCALE_FACTOR) + 50)

def unscale_pos(pixels):
    return ((pixels[0] - 50) / SCALE_FACTOR, (pixels[1] - 50) / SCALE_FACTOR)

def meter_dist(a, b):
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# --- Input Box -----------
# =========================
class InputBox:
    """A class for creating and managing interactive text input boxes."""
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = WHITE
        self.text = text
        self.font = pygame.font.SysFont("Arial", 20)
        self.txt_surface = self.font.render(text, True, BLACK)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.color = LIGHT_BLUE if self.active else WHITE
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode and (event.unicode.isdigit() or event.unicode in '.-'):
                self.text += event.unicode
            self.txt_surface = self.font.render(self.text, True, BLACK)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 1)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

# =========================
# --- Theta* Pathfinding --
# =========================
class Node:
    __slots__ = ("parent", "pos", "g", "h", "f")
    def __init__(self, parent, pos, g=0.0, h=0.0):
        self.parent = parent
        self.pos = pos  # (gx, gy) grid indices
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other):
        return self.f < other.f

def grid_free(grid, gx, gy):
    return 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1] and grid[gx, gy] == 0

def line_of_sight(grid, a, b):
    """Bresenham-style LoS: returns True if straight segment a->b doesn’t cross blocked cells."""
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)

    if dx >= dy:
        err = dx // 2
        y = y0
        for x in range(x0, x1 + sx, sx):
            if not grid_free(grid, x, y):
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
        return True
    else:
        err = dy // 2
        x = x0
        for y in range(y0, y1 + sy, sy):
            if not grid_free(grid, x, y):
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
        return True

def theta_star(grid, start, goal):
    """
    Theta* on 8-connected grid.
    start, goal: integer grid coords (gx, gy)
    Returns path as list of (gx, gy) including both endpoints, or None.
    """
    if start == goal:
        return [start]
    if not grid_free(grid, *start) or not grid_free(grid, *goal):
        return None

    open_heap = []
    nodes = {}
    def h_func(p):
        # Euclidean heuristic (admissible for any-angle motion)
        return math.hypot(p[0]-goal[0], p[1]-goal[1])

    start_node = Node(None, start, g=0.0, h=h_func(start))
    nodes[start] = start_node
    heapq.heappush(open_heap, (start_node.f, start_node))

    closed = set()
    neighbors8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current.pos in closed:
            continue
        closed.add(current.pos)

        if current.pos == goal:
            # reconstruct
            path = []
            n = current
            while n:
                path.append(n.pos)
                n = n.parent
            return list(reversed(path))

        for dx, dy in neighbors8:
            nx, ny = current.pos[0] + dx, current.pos[1] + dy
            if not grid_free(grid, nx, ny):
                continue
            npos = (nx, ny)

            # parent to neighbor shortcut if LoS
            parent = current.parent if current.parent is not None else current
            parent_pos = parent.pos

            if parent is current:
                tentative_g = current.g + math.hypot(dx, dy)
                new_parent = current
            else:
                if line_of_sight(grid, parent_pos, npos):
                    tentative_g = parent.g + math.hypot(npos[0]-parent_pos[0], npos[1]-parent_pos[1])
                    new_parent = parent
                else:
                    tentative_g = current.g + math.hypot(dx, dy)
                    new_parent = current

            if npos not in nodes or tentative_g < nodes[npos].g:
                h = h_func(npos)
                node = Node(new_parent, npos, tentative_g, h)
                nodes[npos] = node
                heapq.heappush(open_heap, (node.f, node))

    return None

def path_grid_to_meters(path_grid, resolution):
    """Convert grid indices path to meter coordinates (cell centers)."""
    if not path_grid:
        return None
    return [(gx / resolution, gy / resolution) for gx, gy in path_grid]

def path_length_meters(path_meters):
    if not path_meters or len(path_meters) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_meters)):
        total += meter_dist(path_meters[i-1], path_meters[i])
    return total

# =========================
# --- Obstacles -----------
# =========================
def create_obstacle_map(resolution=1, table_radius_m=0.6):
    """Creates a grid map with obstacles. 
       We mark table discs of given radius as blocked.
    """
    grid_width = int(math.ceil(RESTAURANT_WIDTH * resolution))
    grid_height = int(math.ceil(RESTAURANT_HEIGHT * resolution))
    grid = np.zeros((grid_width, grid_height), dtype=np.uint8)

    # mark tables
    r_cells = max(1, int(table_radius_m * resolution))
    for table in LAYOUT['tables'].values():
        tx, ty = table['coords']
        gx, gy = int(round(tx * resolution)), int(round(ty * resolution))
        for x in range(gx - r_cells, gx + r_cells + 1):
            for y in range(gy - r_cells, gy + r_cells + 1):
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    if (x - gx)**2 + (y - gy)**2 <= r_cells**2:
                        grid[x, y] = 1
    return grid

# =========================
# --- Movement helpers ----
# =========================
def find_reachable_goal(grid, goal_grid_raw):
    """Pick the goal cell (center or nearest free neighbor) that is free."""
    gx, gy = goal_grid_raw
    if grid_free(grid, gx, gy):
        return (gx, gy)
    # try expanding ring of neighbors
    for radius in (1, 2, 3):
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = gx + dx, gy + dy
                if grid_free(grid, nx, ny):
                    return (nx, ny)
    return goal_grid_raw  # last resort

def animate_follow(env, agent, path_meters, speed_mps):
    """Generic movement along path_meters at speed_mps (updates agent.x/y continuously)."""
    for i in range(1, len(path_meters)):
        start_pos, end_pos = path_meters[i-1], path_meters[i]
        dist = meter_dist(start_pos, end_pos)
        duration = (dist / speed_mps) / 60.0  # minutes

        elapsed = 0.0
        if duration <= 0:
            continue
        while elapsed < duration:
            yield env.timeout(1.0 / FPS / 60.0)  # minutes per frame @ 60fps
            elapsed += 1.0 / FPS / 60.0
            progress = min(1.0, elapsed / duration)
            agent.x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            agent.y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
    agent.x, agent.y = path_meters[-1]

def theta_move(env, agent, start_xy, goal_xy, speed_mps):
    """Compute Theta* path and animate. Fallback to straight-line timing if no path."""
    # We'll compute elapsed time in caller, but log path meta here for diagnostics.
    if OBSTACLE_MAP is None:
        dist = meter_dist(start_xy, goal_xy)
        dur = (dist / speed_mps) / 60.0
        log_event(env, getattr(agent, "id", "agent"), "move_straight",
                  f"dist={dist:.3f}m dur={dur:.3f}min from={start_xy} to={goal_xy}")
        yield env.timeout(dur)
        agent.x, agent.y = goal_xy
        return

    res = OBSTACLE_RES
    start_grid = (int(round(start_xy[0] * res)), int(round(start_xy[1] * res)))
    goal_grid_raw = (int(round(goal_xy[0] * res)), int(round(goal_xy[1] * res)))
    goal_grid = find_reachable_goal(OBSTACLE_MAP, goal_grid_raw)

    path_grid = theta_star(OBSTACLE_MAP, start_grid, goal_grid)
    if not path_grid:
        dist = meter_dist(start_xy, goal_xy)
        dur = (dist / speed_mps) / 60.0
        log_event(env, getattr(agent, "id", "agent"), "move_fallback",
                  f"no_path dist={dist:.3f}m dur={dur:.3f}min start={start_xy} goal={goal_xy}")
        yield env.timeout(dur)
        agent.x, agent.y = goal_xy
        return

    path_meters = path_grid_to_meters(path_grid, res)
    if meter_dist(path_meters[-1], goal_xy) > 1e-6:
        path_meters.append(goal_xy)
    total_len = path_length_meters(path_meters)
    total_dur = (total_len / speed_mps) / 60.0
    log_event(env, getattr(agent, "id", "agent"), "move_theta",
              f"segments={len(path_meters)-1} path_len={total_len:.3f}m dur={total_dur:.3f}min start={start_xy} goal={goal_xy}")
    yield env.process(animate_follow(env, agent, path_meters, speed_mps))

# =========================
# --- Agents --------------
# =========================
class Waiter:
    """Represents a waiter agent in the simulation."""
    def __init__(self, env, id, restaurant):
        self.env, self.id, self.restaurant = env, id, restaurant
        self.x, self.y = LAYOUT["kitchen"]
        all_waiters.append(self)
        log_event(env, f"W{id}", "online", f"at_kitchen={self.x:.2f},{self.y:.2f} speed={WAITER_SPEED}mps")

    def serve_customer(self, table_id):
        """Path to table (neighbor if blocked), serve, return to kitchen."""
        global AVG_WAITER_SERVING_TIME

        res = OBSTACLE_RES
        start_xy = (self.x, self.y)

        table_coords = LAYOUT["tables"][table_id]["coords"]
        goal_grid_raw = (int(round(table_coords[0] * res)), int(round(table_coords[1] * res)))
        end_grid = find_reachable_goal(OBSTACLE_MAP, goal_grid_raw)
        end_xy = (end_grid[0] / res, end_grid[1] / res)

        # Theta* to table
        log_event(self.env, f"W{self.id}", "dispatch_to_table", f"table={table_id} target_xy={end_xy}")
        t0 = self.env.now
        yield self.env.process(theta_move(self.env, self, start_xy, end_xy, WAITER_SPEED))
        t1 = self.env.now
        log_event(self.env, f"W{self.id}", "arrived_table",
                  f"table={table_id} travel_time={(t1 - t0):.3f}min")

        # serving time
        if AVG_WAITER_SERVING_TIME > 0:
            svc = random.expovariate(1.0 / AVG_WAITER_SERVING_TIME)
            log_event(self.env, f"W{self.id}", "service_start", f"table={table_id} svc_draw={svc:.3f}min")
            yield self.env.timeout(svc)
            log_event(self.env, f"W{self.id}", "service_end", f"table={table_id}")
        else:
            log_event(self.env, f"W{self.id}", "service_instant", f"table={table_id}")

        # back to kitchen
        kitchen_xy = LAYOUT["kitchen"]
        t2 = self.env.now
        log_event(self.env, f"W{self.id}", "return_kitchen_begin", f"from_table={table_id}")
        yield self.env.process(theta_move(self.env, self, (self.x, self.y), kitchen_xy, WAITER_SPEED))
        t3 = self.env.now
        log_event(self.env, f"W{self.id}", "return_kitchen_end",
                  f"from_table={table_id} travel_time={(t3 - t2):.3f}min")

        # ready for next task
        log_event(self.env, f"W{self.id}", "available", "ready_for_next_task")
        yield self.restaurant.waiter_store.put(self)

class Customer:
    """Customer uses Theta* for BOTH legs: Entry→Reception and Reception→Table."""
    def __init__(self, env, id, restaurant, target_group):
        self.env, self.id, self.restaurant, self.target_group = env, id, restaurant, target_group
        self.x, self.y = LAYOUT["entry"]
        self.table_id = None
        self.status = 'arriving'
        self.arrival_time = env.now
        all_customers.append(self)
        log_event(env, self.id, "arrival", f"group={self.target_group} at_entry={self.x:.2f},{self.y:.2f}")
        self.env.process(self.run())

    def run(self):
        global AVG_DWELL_TIME, customers_served_count

        # --- 1) Entry -> Reception (Theta*) ---
        reception_xy = LAYOUT["reception"]
        log_event(self.env, self.id, "walk_to_reception_begin", f"from=entry to={reception_xy}")
        t0 = self.env.now
        yield self.env.process(theta_move(self.env, self, (self.x, self.y), reception_xy, CUSTOMER_SPEED))
        t1 = self.env.now
        self.status = 'queuing'
        log_event(self.env, self.id, "at_reception", f"leg_time={(t1 - t0):.3f}min")

        # --- 2) Acquire a table ID for the group ---
        log_event(self.env, self.id, "waiting_for_table", f"group={self.target_group}")
        table_id = yield self.restaurant.table_stores[self.target_group].get()
        self.table_id = table_id

        seating_time = self.env.now
        wait = seating_time - self.arrival_time
        seated_customer_wait_times.append(wait)
        log_event(self.env, self.id, "table_assigned",
            f"table={table_id} total_wait_until_seated={wait:.3f}min")

        # --- 3) Reception -> Table (Theta*) ---
        table_xy = LAYOUT["tables"][self.table_id]["coords"]
        res = OBSTACLE_RES
        goal_grid_raw = (int(round(table_xy[0] * res)), int(round(table_xy[1] * res)))
        end_grid = find_reachable_goal(OBSTACLE_MAP, goal_grid_raw)
        end_xy = (end_grid[0] / res, end_grid[1] / res)

        log_event(self.env, self.id, "walk_to_table_begin", f"table={table_id} to={end_xy}")
        t2 = self.env.now
        yield self.env.process(theta_move(self.env, self, (self.x, self.y), end_xy, CUSTOMER_SPEED))
        t3 = self.env.now
        self.status = 'seated'
        log_event(self.env, self.id, "seated", f"table={table_id} leg_time={(t3 - t2):.3f}min")

        # --- 4) Waiter service (waiter also uses Theta*) ---
        log_event(self.env, self.id, "request_waiter", f"table={table_id}")
        waiter = yield self.restaurant.waiter_store.get()
        log_event(self.env, self.id, "waiter_acquired", f"waiter=W{waiter.id} table={table_id}")
        yield self.env.process(waiter.serve_customer(self.table_id))

        # --- 5) Dwell / Eat ---
        self.status = 'eating'
        if AVG_DWELL_TIME > 0:
            dwell = random.expovariate(1.0 / AVG_DWELL_TIME)
            log_event(self.env, self.id, "dine_start", f"dwell_draw={dwell:.3f}min")
            yield self.env.timeout(dwell)
            log_event(self.env, self.id, "dine_end", f"table={self.table_id}")
        else:
            log_event(self.env, self.id, "dine_instant", "")

        # --- 6) Leave & free table ---
        self.status = 'leaving'
        customers_served_count += 1

        log_event(self.env, self.id, "leaving_table", f"table={self.table_id}")
        yield self.restaurant.table_stores[self.target_group].put(self.table_id)
        log_event(self.env, self.id, "table_released", f"table={self.table_id}")

        # KPI snapshot right after this customer is done
        log_kpis(self.env)

        yield self.env.timeout(0.1)

        # remove from global list
        try:
            all_customers.remove(self)
            log_event(self.env, self.id, "despawn", "")
        except ValueError:
            pass

class Restaurant:
    def __init__(self, env, num_waiters):
        # Stores per table group (TWO groups)
        self.table_stores = {
            'window': simpy.Store(env),
            'aisle':  simpy.Store(env)
        }
        added = {"window": [], "aisle": []}
        for id, props in LAYOUT["tables"].items():
            if props['group'] in self.table_stores:
                self.table_stores[props['group']].put(id)
                added[props['group']].append(id)
        log_event(env, "SYS", "tables_loaded", f"window={added['window']} aisle={added['aisle']}")

        self.waiter_store = simpy.Store(env, capacity=num_waiters)
        for i in range(num_waiters):
            w = Waiter(env, i, self)
            self.waiter_store.put(w)
        log_event(env, "SYS", "waiters_ready", f"num={num_waiters}")

# =========================
# --- Generators ----------
# =========================
def customer_generator(env, restaurant, table_group, arrival_interval_min):
    """
    Deterministic arrivals: one customer every `arrival_interval_min` minutes.
    """
    customer_id = 0
    while True:
        if arrival_interval_min <= 0:
            log_event(env, "GEN", "stop_generator",
                      f"group={table_group} interval={arrival_interval_min} (non-positive)")
            return
        ia = float(arrival_interval_min)
        log_event(env, "GEN", "interarrival_fixed",
                  f"group={table_group} interval_min={ia:.3f}")
        yield env.timeout(ia)
        customer_id += 1
        cid = f"C{table_group[0].upper()}{customer_id}"
        log_event(env, "GEN", "arrival_emit", f"id={cid} group={table_group}")
        Customer(env, cid, restaurant, table_group)

# =========================
# --- Drawing / Grid ------
# =========================
def draw_grid(screen, font_small):
    """1-meter grid with axes & labels."""
    if not SHOW_GRID:
        return
    # Border
    border_rect = pygame.Rect(scale_pos((0, 0)), (RESTAURANT_WIDTH * SCALE_FACTOR, RESTAURANT_HEIGHT * SCALE_FACTOR))
    pygame.draw.rect(screen, BLACK, border_rect, 1)

    # Vertical lines (x)
    x = 0.0
    while x <= RESTAURANT_WIDTH + 1e-6:
        sx, sy = scale_pos((x, 0))
        ex, ey = scale_pos((x, RESTAURANT_HEIGHT))
        pygame.draw.line(screen, LIGHT_GREY, (sx, sy), (ex, ey), 1)
        # x labels along bottom
        label = font_small.render(f"{x:.0f}", True, AXIS_GREY)
        screen.blit(label, (sx - 6, ey + 2))
        x += 1.0

    # Horizontal lines (y)
    y = 0.0
    while y <= RESTAURANT_HEIGHT + 1e-6:
        sx, sy = scale_pos((0, y))
        ex, ey = scale_pos((RESTAURANT_WIDTH, y))
        pygame.draw.line(screen, LIGHT_GREY, (sx, sy), (ex, ey), 1)
        # y labels along left
        label = font_small.render(f"{y:.0f}", True, AXIS_GREY)
        screen.blit(label, (sx - 20, sy - 7))
        y += 1.0

def draw_with_label(screen, font_label, asset_key, coords, label_text):
    img = ASSETS.get(asset_key)
    if not img:
        return
    pos = scale_pos(coords)
    rect = img.get_rect(center=pos)
    screen.blit(img, rect.topleft)
    # label with coordinates
    text = f"{label_text} ({coords[0]:.2f}, {coords[1]:.2f})"
    label = font_label.render(text, True, BLACK)
    screen.blit(label, (rect.centerx - label.get_width()//2, rect.bottom))

def make_fallback_assets():
    """Create simple colored icon surfaces if PNGs are missing."""
    def circle_surf(d, color):
        surf = pygame.Surface((d, d), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (d//2, d//2), d//2)
        pygame.draw.circle(surf, BLACK, (d//2, d//2), d//2, 1)
        return surf
    return {
        'kitchen': circle_surf(36, (255, 220, 180)),
        'reception': circle_surf(32, (220, 255, 220)),
        'entry': circle_surf(28, (220, 220, 255)),
        'table': circle_surf(30, (240, 240, 240)),
        'waiter': circle_surf(24, (255, 240, 0)),
        'customer': circle_surf(22, (100, 180, 250)),
    }

# =========================
# --- Demo Validator ------
# =========================
DEMO_LAYOUTS = [
    # Each: name, (W,H), kitchen(x,y), target(x,y), description
    ("A", (20.0, 15.0), (2.0, 2.0), (10.0, 2.0),  "Horizontal straight"),
    ("B", (20.0, 15.0), (2.0, 2.0), (2.0, 12.0),  "Vertical straight"),
    ("C", (20.0, 15.0), (1.0, 1.0), (19.0, 14.0), "General diagonal"),
    ("D", (20.0, 15.0), (5.0, 5.0), (15.0, 10.0), "General oblique"),
    ("E", (20.0, 15.0), (0.5, 7.5), (19.5, 7.5),  "Edge-to-edge horizontal"),
]

def validate_demo_layouts():
    """
    Uses Theta* with EMPTY obstacle map, checks Kitchen->Target->Kitchen
    against mathematical ground truth: 2 * hypot(dx, dy).
    Stores results in DEMO_RESULTS and prints them.
    """
    global DEMO_RESULTS
    DEMO_RESULTS = []
    resolution = 1  # with LoS any-angle, resolution doesn't affect the straight free-space path

    for name, (W, H), K, T, note in DEMO_LAYOUTS:
        # empty grid
        gw = int(math.ceil(W * resolution))
        gh = int(math.ceil(H * resolution))
        grid = np.zeros((gw, gh), dtype=np.uint8)

        start = (int(round(K[0]*resolution)), int(round(K[1]*resolution)))
        goal  = (int(round(T[0]*resolution)), int(round(T[1]*resolution)))

        path1 = theta_star(grid, start, goal)
        path2 = theta_star(grid, goal, start)

        def pm(pl): 
            return [(x/resolution, y/resolution) for (x,y) in pl] if pl else None
        pm1 = pm(path1); pm2 = pm(path2)

        def plen(p):
            if not p or len(p) < 2: return 0.0
            s=0.0
            for i in range(1,len(p)): s += meter_dist(p[i-1], p[i])
            return s
        path_len = plen(pm1) + plen(pm2)

        dx = T[0] - K[0]
        dy = T[1] - K[1]
        gt_roundtrip = 2.0 * math.hypot(dx, dy)

        DEMO_RESULTS.append({
            "Layout": name,
            "Note": note,
            "Kitchen": K,
            "Target": T,
            "GT Round (m)": gt_roundtrip,
            "Path Round (m)": path_len,
            "Δ (m)": path_len - gt_roundtrip
        })

    # Print to console neatly
    print("\n=== DEMO VALIDATOR (Kitchen -> Target -> Kitchen) ===")
    print(f"{'Layout':<8}{'Note':<24}{'GT Round (m)':>14}{'Path Round (m)':>16}{'Δ (m)':>12}")
    for r in DEMO_RESULTS:
        print(f"{r['Layout']:<8}{r['Note']:<24}{r['GT Round (m)']:>14.6f}{r['Path Round (m)']:>16.6f}{r['Δ (m)']:>12.6f}")
    print("Expected: Δ ~ 0.000000 for all rows (any-angle Theta* in free space).\n")

# =========================
# --- Main ----------------
# =========================
def main():
    global NUM_WAITERS, WAITER_SPEED, CUSTOMER_SPEED, SIM_SPEED_FACTOR, LAYOUT, ASSETS, AVG_WAITER_SERVING_TIME, AVG_DWELL_TIME
    global RESTAURANT_WIDTH, RESTAURANT_HEIGHT, OBSTACLE_MAP, OBSTACLE_RES, SHOW_GRID, SHOW_HELP

    pygame.init()
    init_logger()  # initialize the loggers
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 16, True)
    font_title = pygame.font.SysFont("Arial", 28, True)
    font_label = pygame.font.SysFont("Arial", 12)
    font_small = pygame.font.SysFont("Arial", 10)

    # Assets (fallback if images missing)
    try:
        ASSETS = {
            'kitchen': pygame.transform.scale(pygame.image.load('kitchen.png'), (40, 40)),
            'reception': pygame.transform.scale(pygame.image.load('reception.png'), (35, 35)),
            'entry': pygame.transform.scale(pygame.image.load('entry.png'), (30, 30)),
            'table': pygame.transform.scale(pygame.image.load('table.png'), (35, 35)),
            'waiter': pygame.transform.scale(pygame.image.load('waiter.png'), (30, 30)),
            'customer': pygame.transform.scale(pygame.image.load('customer.png'), (25, 25)),
        }
    except Exception:
        ASSETS = make_fallback_assets()

    reset_simulation_state()
    app_state, placement_mode, table_id_counter = 'SETUP_LAYOUT', 'k', 1
    update_scale_factor()

    input_boxes = {
        'width':        InputBox(CANVAS_WIDTH + 25, 120, 200, 32, str(RESTAURANT_WIDTH)),
        'height':       InputBox(CANVAS_WIDTH + 25, 190, 200, 32, str(RESTAURANT_HEIGHT)),
        'num_waiters':  InputBox(CANVAS_WIDTH + 25, 260, 200, 32, str(NUM_WAITERS)),
        'waiter_speed': InputBox(CANVAS_WIDTH + 25, 330, 200, 32, str(WAITER_SPEED)),
        'sim_speed':    InputBox(CANVAS_WIDTH + 25, 400, 200, 32, str(SIM_SPEED_FACTOR)),
        'serving_time': InputBox(CANVAS_WIDTH + 25, 470, 200, 32, str(AVG_WAITER_SERVING_TIME)),
        'dwell_time':   InputBox(CANVAS_WIDTH + 25, 540, 200, 32, str(AVG_DWELL_TIME)),
        'window_rate':  InputBox(CANVAS_WIDTH + 25, 610, 200, 32, '8'),
        'aisle_rate':   InputBox(CANVAS_WIDTH + 25, 680, 200, 32, '15')
    }

    running, env = True, None
    restaurant = None

    # Help text
    help_lines = [
        "[E] Entry   [K] Kitchen   [R] Reception   [T] Window Table   [Y] Aisle Table",
        "Click on canvas to place selected element. ENTER to proceed.",
        "[G] Toggle Grid   [H] Toggle Help   [D] Run Demo Validator (compare math vs code)",
        "RUNNING mode: BACKSPACE to reset layout"
    ]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Global hotkeys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g: SHOW_GRID = not SHOW_GRID
                if event.key == pygame.K_h: SHOW_HELP = not SHOW_HELP
                if event.key == pygame.K_d:
                    validate_demo_layouts()

            if app_state == 'SETUP_LAYOUT':
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_e, pygame.K_k, pygame.K_r, pygame.K_t, pygame.K_y):
                        placement_mode = {pygame.K_e: 'e', pygame.K_k: 'k', pygame.K_r: 'r',
                                          pygame.K_t: 'window', pygame.K_y: 'aisle'}[event.key]
                    if event.key == pygame.K_RETURN and all(LAYOUT[key] for key in ['kitchen', 'reception', 'entry']):
                        app_state = 'SETUP_PARAMETERS'
                if event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < CANVAS_WIDTH:
                    coords = unscale_pos(event.pos)
                    # keep within bounds
                    coords = (clamp(coords[0], 0, RESTAURANT_WIDTH), clamp(coords[1], 0, RESTAURANT_HEIGHT))
                    if placement_mode == 'e':
                        LAYOUT['entry'] = coords
                        log_event(env or type('E', (), {'now':0})(), "UI", "place_entry", f"{coords}")
                    elif placement_mode == 'k':
                        LAYOUT['kitchen'] = coords
                        log_event(env or type('E', (), {'now':0})(), "UI", "place_kitchen", f"{coords}")
                    elif placement_mode == 'r':
                        LAYOUT['reception'] = coords
                        log_event(env or type('E', (), {'now':0})(), "UI", "place_reception", f"{coords}")
                    else:
                        LAYOUT['tables'][table_id_counter] = {"coords": coords, "group": placement_mode}
                        log_event(env or type('E', (), {'now':0})(), "UI", "place_table",
                                  f"id={table_id_counter} group={placement_mode} coords={coords}")
                        table_id_counter += 1

            elif app_state == 'SETUP_PARAMETERS':
                for box in input_boxes.values():
                    box.handle_event(event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    try:
                        RESTAURANT_WIDTH  = float(input_boxes['width'].text)
                        RESTAURANT_HEIGHT = float(input_boxes['height'].text)
                        NUM_WAITERS       = int(input_boxes['num_waiters'].text)
                        WAITER_SPEED      = float(input_boxes['waiter_speed'].text)
                        CUSTOMER_SPEED    = WAITER_SPEED * 1.5
                        SIM_SPEED_FACTOR  = float(input_boxes['sim_speed'].text)
                        AVG_WAITER_SERVING_TIME = float(input_boxes['serving_time'].text)
                        AVG_DWELL_TIME    = float(input_boxes['dwell_time'].text)
                        LAYOUT['arrival_rates']['window'] = float(input_boxes['window_rate'].text)
                        LAYOUT['arrival_rates']['aisle']  = float(input_boxes['aisle_rate'].text)

                        update_scale_factor()
                        OBSTACLE_MAP = create_obstacle_map(resolution=OBSTACLE_RES)
                        app_state, env = 'RUNNING', simpy.Environment()
                        pygame.display.set_caption("Restaurant Simulation")
                        restaurant = Restaurant(env, NUM_WAITERS)
                        for group, rate in LAYOUT["arrival_rates"].items():
                            has_tables = any(t['group'] == group for t in LAYOUT['tables'].values())
                            log_event(env, "SYS", "generator_start",
                                      f"group={group} interval_min={rate} tables_present={has_tables}")
                            if rate > 0 and has_tables:
                                env.process(customer_generator(env, restaurant, group, rate))
                        log_kpis(env)
                    except (ValueError, TypeError):
                        print("Invalid input.")

            elif app_state == 'RUNNING':
                if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                    log_event(env, "SYS", "reset_requested", "")
                    reset_simulation_state()
                    app_state, table_id_counter = 'SETUP_LAYOUT', 1
                    update_scale_factor()
                    pygame.display.set_caption("Restaurant Simulation Setup")

        # ----- DRAW -----
        screen.fill(WHITE)
        pygame.draw.rect(screen, GREY, (CANVAS_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(screen, BLACK, (CANVAS_WIDTH, 0), (CANVAS_WIDTH, SCREEN_HEIGHT), 2)

        # Grid / axes
        draw_grid(screen, font_small)

        # Side bar title
        if app_state == 'SETUP_LAYOUT':
            screen.blit(font_title.render("Layout Setup", True, BLACK), (CANVAS_WIDTH + 35, 20))
            labels = {
                "[E] Entry": 80,
                "[K] Kitchen": 120,
                "[R] Reception": 160,
                "[T] Window Table": 200,
                "[Y] Aisle Table": 240
            }
            y_help = 280
            for text, y_pos in labels.items():
                screen.blit(font_bold.render(text, True, BLACK), (CANVAS_WIDTH + 25, y_pos))

            if SHOW_HELP:
                help_lines = [
                    "[E] Entry   [K] Kitchen   [R] Reception   [T] Window Table   [Y] Aisle Table",
                    "Click on canvas to place selected element. ENTER to proceed.",
                    "[G] Toggle Grid   [H] Toggle Help   [D] Run Demo Validator (compare math vs code)",
                    "RUNNING mode: BACKSPACE to reset layout"
                ]
                for i, line in enumerate(help_lines):
                    screen.blit(font_small.render(line, True, BLUE), (CANVAS_WIDTH + 10, y_help + i*16))

            screen.blit(font_bold.render("Press ENTER to Continue", True, BLUE), (CANVAS_WIDTH + 20, SCREEN_HEIGHT - 50))

        elif app_state == 'SETUP_PARAMETERS':
            screen.blit(font_title.render("Parameters", True, BLACK), (CANVAS_WIDTH + 50, 20))
            labels = {
                'width': "Restaurant Width (m):",
                'height': "Restaurant Height (m):",
                'num_waiters': "Number of Waiters:",
                'waiter_speed': "Waiter Speed (m/s):",
                'sim_speed': "Sim Speed (x times):",
                'serving_time': "Avg. Serving Time (mins):",
                'dwell_time': "Avg. Dwell Time (mins):",
                'window_rate': "Window Arrival (mins):",
                'aisle_rate': "Aisle Arrival (mins):"
            }
            for key, box in input_boxes.items():
                screen.blit(font_bold.render(labels.get(key, ""), True, BLACK), (box.rect.x, box.rect.y - 20))
                box.draw(screen)

            if SHOW_HELP:
                help_lines = [
                    "Theta* is used for both customers and waiters around table obstacles.",
                    "Place entry/kitchen/reception and at least one table per group.",
                    "Arrivals occur every N minutes (deterministic)."
                ]
                for i, line in enumerate(help_lines):
                    screen.blit(font_small.render(line, True, BLUE), (CANVAS_WIDTH + 10, 10 + i*16 + 550))

            screen.blit(font_bold.render("Press ENTER to Start", True, BLUE), (CANVAS_WIDTH + 35, SCREEN_HEIGHT - 50))

        # Draw placed items + coordinates
        if LAYOUT.get("entry"):
            draw_with_label(screen, font_label, 'entry', LAYOUT["entry"], "Entry")
        if LAYOUT.get("kitchen"):
            draw_with_label(screen, font_label, 'kitchen', LAYOUT["kitchen"], "Kitchen")
        if LAYOUT.get("reception"):
            draw_with_label(screen, font_label, 'reception', LAYOUT["reception"], "Reception")
        for id, table in LAYOUT["tables"].items():
            draw_with_label(screen, font_label, 'table', table["coords"], f"T{id}")

        # Running: KPIs + agents
        if app_state == 'RUNNING':
            # Advance env by 1/FPS minutes scaled by SIM_SPEED_FACTOR
            env.run(until=env.now + (1.0 / FPS * SIM_SPEED_FACTOR))

            # queue visuals (line to the right of reception)
            if LAYOUT.get("reception"):
                queue_position_start = np.array(LAYOUT["reception"]) + np.array([2.0, 0.0])
                queue_idx = 0
                for customer in all_customers:
                    if customer.status == 'queuing':
                        customer.x, customer.y = (queue_position_start + np.array([queue_idx * 1.5, 0])).tolist()
                        queue_idx += 1

            # Draw customers
            for customer in all_customers:
                draw_with_label(screen, font_label, 'customer', (customer.x, customer.y), customer.id)

            # Draw waiters
            for waiter in all_waiters:
                screen.blit(ASSETS['waiter'], ASSETS['waiter'].get_rect(center=scale_pos((waiter.x, waiter.y))))
                wlabel = font_label.render(f"W{waiter.id} ({waiter.x:.2f},{waiter.y:.2f})", True, BLACK)
                wpos = scale_pos((waiter.x, waiter.y))
                screen.blit(wlabel, (wpos[0] - wlabel.get_width()//2, wpos[1] + 14))

            # KPIs
            screen.blit(font_title.render("Live KPIs", True, BLACK), (CANVAS_WIDTH + 50, 20))
            backlog_customers = [c for c in all_customers if c.status == 'queuing']
            active_customers = [c for c in all_customers if c.status in ['seated', 'eating']]
            avg_seated_wait = np.mean(seated_customer_wait_times) if seated_customer_wait_times else 0.0
            current_backlog_waits = [env.now - c.arrival_time for c in backlog_customers] if env else []
            avg_backlog_wait = np.mean(current_backlog_waits) if current_backlog_waits else 0.0

            kpis = [
                f"Customers Served: {customers_served_count}",
                "--------------------",
                f"Active Customers: {len(active_customers)}",
                f"Backlog Customers: {len(backlog_customers)}",
                "--------------------",
                f"Avg Seated Wait: {avg_seated_wait:.1f} min",
                f"Avg Backlog Wait: {avg_backlog_wait:.1f} min",
                "--------------------",
                f"Sim Time: {int(env.now // 60)}h {int(env.now % 60)}m" if env else "Sim Time: 0h 0m"
            ]
            y_offset = 80
            for text in kpis:
                screen.blit(font_bold.render(text, True, BLACK), (CANVAS_WIDTH + 20, y_offset))
                y_offset += 28

            # Inputs recap
            params_title = font_bold.render("INPUT PARAMETERS:", True, BLACK)
            screen.blit(params_title, (CANVAS_WIDTH + 20, 350))
            params = [
                f"Avg Dwell Time: {AVG_DWELL_TIME:.1f} min",
                f"Avg Serving Time: {AVG_WAITER_SERVING_TIME:.1f} min",
                f"Window Arrival: {LAYOUT.get('arrival_rates', {}).get('window', 0):.1f} min",
                f"Aisle Arrival: {LAYOUT.get('arrival_rates', {}).get('aisle', 0):.1f} min"
            ]
            y_offset = 380
            for text in params:
                screen.blit(font_bold.render(text, True, BLUE), (CANVAS_WIDTH + 20, y_offset))
                y_offset += 24

            screen.blit(font_bold.render("Press BACKSPACE to Restart", True, RED), (CANVAS_WIDTH + 5, SCREEN_HEIGHT - 50))

        # Demo results overlay (if ran)
        if DEMO_RESULTS:
            screen.blit(font_bold.render("DEMO RESULTS", True, BLACK), (CANVAS_WIDTH + 35, 220))
            y = 242
            for r in DEMO_RESULTS:
                line = f"{r['Layout']}: GT {r['GT Round (m)']:.3f} | Path {r['Path Round (m)']:.3f} | Δ {r['Δ (m)']:.3f}"
                screen.blit(font_small.render(line, True, (0,100,0) if abs(r['Δ (m)']) < 1e-6 else RED), (CANVAS_WIDTH + 10, y))
                y += 16

        # bottom-left dimensions
        dim_text = font_bold.render(f"Dimensions: {RESTAURANT_WIDTH:.1f}m x {RESTAURANT_HEIGHT:.1f}m", True, BLACK)
        screen.blit(dim_text, (10, SCREEN_HEIGHT - 30))

        pygame.display.flip()
        flush_log_to_file()  # keep CSV up to date each frame
        clock.tick(FPS)

    flush_log_to_file()
    pygame.quit()

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()
