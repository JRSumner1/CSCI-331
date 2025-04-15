import heapq  # Module for heap queue implementation (priority queue)
from math import sqrt  # Module for mathematical functions
from sys import argv  # Import command-line arguments
from PIL import Image  # Python Imaging Library for image processing

# Define terrain types and their corresponding movement speeds
TERRAIN_SPEEDS = {
    (71, 51, 3): 0.5,    # Paved road - easiest
    (0, 0, 0): 0.6,      # Footpath
    (248, 148, 18): 1.0, # Open land
    (255, 255, 255): 1.2, # Easy movement forest
    (255, 192, 0): 1.5,  # Rough meadow
    (2, 208, 60): 2.0,   # Slow run forest
    (2, 136, 40): 3.0,   # Walk forest
    (0, 0, 255): 8.0,    # Lake/Swamp/Marsh - hardest
    (5, 73, 24): float('inf'), # Impassible vegetation
    (205, 0, 101): float('inf')  # Out of bounds
}

# Conversion factors from pixels to meters
pixel_X = 10.29
pixel_Y = 7.55

# Load terrain map from an image file
def load_terrain_map(filename):
    img = Image.open(filename).convert('RGB')  # Open and convert image to RGB mode
    pixels = img.load()  # Load pixel data from the image
    width, height = img.size  # Get image dimensions (width and height)
    return pixels, width, height

# Load elevation map from a text file
def load_elevation_map(filename):
    elevations = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()  # Split line into values
            # Ignore the last 5 values
            elevations.append(list(map(float, values[:-5])))
    return elevations

# Load path points from a text file
def load_path_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(int, line.split())  # Parse the x and y coordinates
            points.append((x, y))  # Add the point to the list
    return points

# Calculate Euclidean distance between two points, taking into account elevation change
def heuristic(p1, p2, elevations):
    dx = (p1[0] - p2[0]) * pixel_X # Horizontal distance between point 1 and point 2 in meters
    dy = (p1[1] - p2[1]) * pixel_Y # Vertical distance between point 1 and point 2 in meters
    elevation_change = abs(elevations[p2[1]][p2[0]] - elevations[p1[1]][p1[0]]) # Absolute value of elevation between the 2 points
    return sqrt(dx ** 2 + (dy + elevation_change) ** 2) # Adding elevation_change to vertical distance

# Perform A* search algorithm to find the shortest path from start to goal
def a_star_search(start, goal, terrain, elevations, width, height):
    open_set = []  # Priority queue to keep track of nodes to explore
    heapq.heappush(open_set, (0, start))  # Initialize the open set with the start node
    came_from = {}  # Dictionary to track the path to each node
    g_score = {start: 0}  # Cost from start to each node
    f_score = {start: heuristic(start, goal, elevations)}  # Estimated total cost to goal

    while open_set:
        _, current = heapq.heappop(open_set)  # Get the node with the lowest f_score
        
        # Check if the goal has been reached
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path  # Return the path from start to goal

        x, y = current
        # Explore 8 neighboring pixels (including diagonals)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next_x, next_y = x + dx, y + dy
            # Check if the neighboring pixel is within bounds
            if 0 <= next_x < width and 0 <= next_y < height:
                # Terrain Cost Calculation
                terrain_color = terrain[next_x, next_y]  # Get the color of the neighboring pixel
                terrain_speed = TERRAIN_SPEEDS.get(terrain_color, float('inf'))  # Get movement speed, returns 'inf' by default
                step_cost = heuristic((0, 0), (dx, dy), elevations) * terrain_speed  # Cost of moving to neighbor

                temp_g_score = g_score[current] + step_cost  # New cost from start to neighbor
                neighbor = (next_x, next_y)
                # Check if the new path is better
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, elevations)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # Return an empty list if no path is found

# Draw the computed path and control points on the original image and save new image
def draw_path(image, control_points, path, output_path):
    img = Image.open(image).convert('RGBA')  # Open image in RGBA mode
    pixels = img.load()

    # Draw the path
    for (x, y) in path:
        if 0 <= x < img.size[0] and 0 <= y < img.size[1]:
            pixels[x, y] = (118, 63, 231)  # Set path color to purple
    
    # Draw the control points
    for (x, y) in control_points:
        if 0 <= x < img.size[0] and 0 <= y < img.size[1]:
            pixels[x, y] = (255, 0, 0)  # Set control points color to red

    img.save(output_path)  # Save the modified image

def main():
    # Get file paths from command-line arguments
    terrain_file, elevation_file, path_file, output_file = argv[1:]

    # Load data
    terrain, width, height = load_terrain_map(terrain_file)
    elevations = load_elevation_map(elevation_file)
    points = load_path_file(path_file)

    # Initialize variables for A* search results
    total_distance = 0  # To store the total distance of the path
    path = []  # To store the final path

    # Compute the path segment by segment
    for i in range(len(points) - 1):
        start = points[i]
        goal = points[i + 1]  # Next point in the path

        segment_path = a_star_search(start, goal, terrain, elevations, width, height)  # Perform A* search

        # Append the returned path and add to total distance
        if segment_path:
            path.extend(segment_path)
            total_distance += sum(heuristic(segment_path[j], segment_path[j + 1], elevations) for j in range(len(segment_path) - 1))
    
    # Output results
    draw_path(terrain_file, points, path, output_file)
    print(f"Total Distance: {total_distance} m")

if __name__ == "__main__":
    main()  # Execute the main function
