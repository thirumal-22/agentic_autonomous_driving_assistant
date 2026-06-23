#!/usr/bin/env python3
"""
Create a professional logo for the Agentic Autonomous Driving Assistant
"""

from PIL import Image, ImageDraw
import math

def create_logo(width=300, height=300, output_path='assets/logo.png'):
    """Create a professional autonomous driving assistant logo"""
    
    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Define colors (cyberpunk theme)
    primary_color = (0, 255, 200)  # Cyan
    secondary_color = (255, 0, 128)  # Magenta
    accent_color = (100, 200, 255)  # Light Blue
    dark_color = (15, 17, 26)  # Dark background
    
    center_x, center_y = width // 2, height // 2
    
    # Draw circular background with gradient effect (simulated with circles)
    for i in range(15, 0, -1):
        radius = (width // 3) * (i / 15)
        alpha = int(30 * (i / 15))
        color = (*primary_color, alpha)
        draw.ellipse(
            [(center_x - radius, center_y - radius), 
             (center_x + radius, center_y + radius)],
            fill=color,
            outline=None
        )
    
    # Draw agent network nodes (5 nodes in pentagon pattern)
    num_nodes = 5
    node_radius = width // 3.5
    node_size = 12
    
    node_positions = []
    for i in range(num_nodes):
        angle = (i * 2 * math.pi / num_nodes) - (math.pi / 2)
        x = center_x + node_radius * math.cos(angle)
        y = center_y + node_radius * math.sin(angle)
        node_positions.append((x, y))
    
    # Draw connections between nodes
    for i in range(num_nodes):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[(i + 1) % num_nodes]
        draw.line([(x1, y1), (x2, y2)], fill=(*accent_color, 200), width=2)
        
        # Draw diagonal connections for more connectivity
        if i < num_nodes - 2:
            x3, y3 = node_positions[i + 2]
            draw.line([(x1, y1), (x3, y3)], fill=(*accent_color, 100), width=1)
    
    # Draw nodes
    for i, (x, y) in enumerate(node_positions):
        # Node circle
        draw.ellipse(
            [(x - node_size, y - node_size), (x + node_size, y + node_size)],
            fill=secondary_color if i % 2 == 0 else primary_color,
            outline=(*primary_color, 255),
            width=2
        )
        
        # Inner glow
        draw.ellipse(
            [(x - node_size//2, y - node_size//2), (x + node_size//2, y + node_size//2)],
            fill=(*primary_color, 150),
            outline=None
        )
    
    # Draw central autonomous vehicle icon (car silhouette with AI symbol)
    # Central circle with AI gradient
    center_circle_size = 35
    draw.ellipse(
        [(center_x - center_circle_size, center_y - center_circle_size),
         (center_x + center_circle_size, center_y + center_circle_size)],
        fill=(*primary_color, 200),
        outline=(*secondary_color, 255),
        width=3
    )
    
    # Draw a stylized car/AI symbol in the center
    # Draw neural network pattern inside
    for dx in [-15, 0, 15]:
        for dy in [-15, 0, 15]:
            if dx != 0 or dy != 0:
                draw.ellipse(
                    [(center_x + dx - 2, center_y + dy - 2),
                     (center_x + dx + 2, center_y + dy + 2)],
                    fill=(*dark_color, 255),
                    outline=None
                )
    
    # Draw directional arrow (forward/movement)
    arrow_length = 15
    arrow_width = 3
    # Arrow shaft
    draw.line(
        [(center_x, center_y - 25), (center_x, center_y - 40)],
        fill=(*secondary_color, 255),
        width=arrow_width
    )
    # Arrow head
    draw.polygon(
        [(center_x, center_y - 40),
         (center_x - 6, center_y - 32),
         (center_x + 6, center_y - 32)],
        fill=(*secondary_color, 255)
    )
    
    # Save logo
    img.save(output_path, 'PNG')
    print(f"Logo created successfully: {output_path}")
    return img

if __name__ == "__main__":
    create_logo()
