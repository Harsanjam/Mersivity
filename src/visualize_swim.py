#!/usr/bin/env python3
"""
Milestone 3: SWIM-Inspired Real-Time Multimodal Feedback Visualizer

This module provides synchronized audio playback and geometric flow visualization
driven by EEG band features (alpha, beta, theta).

Features:
- SWIM-inspired particle flow system with trails
- Geometric patterns driven by EEG bands
- Synchronized audio playback
- Pseudo-real-time playback from pre-extracted features
"""

import pygame
import numpy as np
import pandas as pd
import soundfile as sf
import time
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class ParticleTrail:
    """Particle with trail for SWIM-inspired visualization"""
    x: float
    y: float
    vx: float
    vy: float
    trail_points: List[Tuple[float, float]]
    max_trail_length: int = 30
    color: Tuple[int, int, int] = (255, 255, 255)
    size: float = 3.0
    
    def update(self, dt: float, force_x: float = 0.0, force_y: float = 0.0):
        """Update particle position and trail"""
        # Apply forces
        self.vx += force_x * dt
        self.vy += force_y * dt
        
        # Damping
        self.vx *= 0.98
        self.vy *= 0.98
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Add current position to trail
        self.trail_points.append((self.x, self.y))
        
        # Limit trail length
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
    
    def draw(self, screen: pygame.Surface):
        """Draw particle and its trail"""
        # Draw trail with fading effect
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                alpha = int(255 * (i / len(self.trail_points)))
                color = tuple(int(c * alpha / 255) for c in self.color)
                
                start_pos = (int(self.trail_points[i-1][0]), int(self.trail_points[i-1][1]))
                end_pos = (int(self.trail_points[i][0]), int(self.trail_points[i][1]))
                
                # Draw trail segment with thickness proportional to alpha
                thickness = max(1, int(self.size * alpha / 255))
                try:
                    pygame.draw.line(screen, color, start_pos, end_pos, thickness)
                except (ValueError, TypeError):
                    pass  # Handle out-of-bounds positions
        
        # Draw particle
        try:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))
        except (ValueError, TypeError):
            pass


class SWIMVisualizer:
    """SWIM-inspired multimodal feedback visualizer"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        pygame.mixer.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mersivity - SWIM Multimodal Feedback")
        
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Color schemes for calm vs stress
        self.calm_colors = [
            (100, 150, 255),  # Soft blue
            (150, 200, 255),  # Light blue
            (200, 220, 255),  # Pale blue
        ]
        
        self.stress_colors = [
            (255, 100, 100),  # Red
            (255, 150, 50),   # Orange
            (255, 200, 100),  # Yellow
        ]
        
       
        
        # Audio playback
        self.audio_data = None
        self.audio_samplerate = None
        self.audio_start_time = None
        
        # Feature data
        self.features_df = None
        self.current_feature_idx = 0
        self.feature_start_time = None
        
        # Current EEG state
        self.alpha_power = 0.5
        self.beta_power = 0.5
        self.theta_power = 0.5
        self.label = "calm"
        
        # Geometric flow parameters
        self.rotation_angle = 0.0
        self.flow_radius = 200.0
        self.spiral_tightness = 0.1

         # Initialize particles
        self.particles: List[ParticleTrail] = []
        self.initialize_particles(num_particles=50)
        
    def initialize_particles(self, num_particles: int = 50):
        """Initialize particle system"""
        self.particles = []
        for i in range(num_particles):
            angle = (i / num_particles) * 2 * math.pi
            radius = self.flow_radius
            
            x = self.width / 2 + radius * math.cos(angle)
            y = self.height / 2 + radius * math.sin(angle)
            
            vx = 0.0
            vy = 0.0
            
            color = self.calm_colors[i % len(self.calm_colors)]
            
            particle = ParticleTrail(
                x=x, y=y, vx=vx, vy=vy,
                trail_points=[],
                max_trail_length=30,
                color=color,
                size=3.0
            )
            self.particles.append(particle)
    
    def load_audio(self, wav_path: str):
        """Load sonification audio"""
        self.audio_data, self.audio_samplerate = sf.read(wav_path)
        print(f"Loaded audio: {len(self.audio_data)} samples @ {self.audio_samplerate} Hz")
        
    def load_features(self, csv_path: str):
        """Load extracted features"""
        self.features_df = pd.read_csv(csv_path)
        print(f"Loaded features: {len(self.features_df)} windows")
        print(f"Columns: {list(self.features_df.columns)}")
        
    def start_playback(self):
        """Start synchronized audio and visualization playback"""
        if self.audio_data is None:
            print("Warning: No audio loaded")
            return
        
        # Start audio playback using pygame mixer
        # Convert to format pygame expects
        audio = self.audio_data

        # Ensure stereo for pygame mixer
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)

        audio_normalized = (audio * 32767).astype(np.int16)

        sound = pygame.sndarray.make_sound(audio_normalized)
        sound.play()
        
        # Record start time
        self.audio_start_time = time.time()
        self.feature_start_time = time.time()
        self.current_feature_idx = 0
        
        print("Playback started!")
        
    def update_features_from_time(self, current_time: float):
        """Update current feature values based on elapsed time"""
        if self.features_df is None or len(self.features_df) == 0:
            return
        
        # Find the appropriate feature window based on time
        elapsed = current_time - self.feature_start_time
        
        # Find closest window
        for idx, row in self.features_df.iterrows():
            t_start = row['t_start']
            t_end = row['t_end']
            
            if t_start <= elapsed <= t_end:
                self.current_feature_idx = idx
                
                # Update feature values (normalized 0-1)
                self.alpha_power = row.get('bp_alpha_norm', 0.5)
                self.beta_power = row.get('bp_beta_norm', 0.5)
                self.theta_power = row.get('bp_theta_norm', 0.5)
                
                # Update label if available
                label_str = str(row.get('label', '')).strip().lower()
                if label_str in ['calm', 'stress', 'stressed']:
                    self.label = 'stress' if label_str in ['stress', 'stressed'] else 'calm'
                
                break
    
    def update_particles(self, dt: float):
        """Update particle system based on current EEG state"""
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Rotation speed controlled by alpha (calm/meditation)
        rotation_speed = 0.5 + 1.5 * self.alpha_power  # 0.5 to 2.0 rad/s
        self.rotation_angle += rotation_speed * dt
        
        # Flow radius controlled by beta (arousal/stress)
        self.flow_radius = 150 + 150 * self.beta_power  # 150 to 300
        
        # Spiral tightness controlled by theta (relaxation)
        self.spiral_tightness = 0.05 + 0.15 * self.theta_power  # 0.05 to 0.2
        
        # Update particle colors based on state
        target_colors = self.stress_colors if self.label == 'stress' else self.calm_colors
        
        for i, particle in enumerate(self.particles):
            # Calculate target position in rotary flow
            angle_offset = (i / len(self.particles)) * 2 * math.pi
            current_angle = self.rotation_angle + angle_offset
            
            # Spiral effect
            current_radius = self.flow_radius * (1.0 + self.spiral_tightness * math.sin(current_angle * 2))
            
            target_x = center_x + current_radius * math.cos(current_angle)
            target_y = center_y + current_radius * math.sin(current_angle)
            
            # Apply force toward target position
            dx = target_x - particle.x
            dy = target_y - particle.y
            
            force_magnitude = 100.0 + 200.0 * self.beta_power  # Stronger forces for high beta
            force_x = dx * force_magnitude * 0.001
            force_y = dy * force_magnitude * 0.001
            
            # Update particle
            particle.update(dt, force_x, force_y)
            
            # Wrap around screen edges
            if particle.x < 0:
                particle.x = self.width
            elif particle.x > self.width:
                particle.x = 0
            
            if particle.y < 0:
                particle.y = self.height
            elif particle.y > self.height:
                particle.y = 0
            
            # Update color (smooth transition)
            target_color = target_colors[i % len(target_colors)]
            current_color = particle.color
            
            # Lerp toward target color
            blend_factor = 0.05
            new_color = tuple(
                int(current_color[j] * (1 - blend_factor) + target_color[j] * blend_factor)
                for j in range(3)
            )
            particle.color = new_color
            
            # Update trail length based on theta (longer trails when relaxed)
            particle.max_trail_length = int(20 + 40 * self.theta_power)
            
            # Update size based on alpha (larger when calm/meditative)
            particle.size = 2.0 + 3.0 * self.alpha_power
    
    def draw_background(self):
        """Draw background with subtle gradient"""
        if self.label == 'stress':
            # Warm background for stress
            top_color = (40, 20, 20)
            bottom_color = (60, 30, 30)
        else:
            # Cool background for calm
            top_color = (20, 20, 40)
            bottom_color = (30, 30, 60)
        
        for y in range(self.height):
            blend = y / self.height
            color = tuple(
                int(top_color[i] * (1 - blend) + bottom_color[i] * blend)
                for i in range(3)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
    
    def draw_info(self):
        """Draw information overlay"""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Title
        title_text = font.render("Mersivity SWIM Visualization", True, (255, 255, 255))
        self.screen.blit(title_text, (20, 20))
        
        # Current state
        state_text = small_font.render(f"State: {self.label.upper()}", True, (255, 255, 255))
        self.screen.blit(state_text, (20, 60))
        
        # Feature bars
        bar_x = 20
        bar_y = 100
        bar_width = 200
        bar_height = 20
        bar_spacing = 35
        
        features = [
            ("Alpha (Meditation)", self.alpha_power, (100, 150, 255)),
            ("Beta (Arousal)", self.beta_power, (255, 150, 100)),
            ("Theta (Relaxation)", self.theta_power, (150, 255, 150)),
        ]
        
        for i, (name, value, color) in enumerate(features):
            y = bar_y + i * bar_spacing
            
            # Label
            label = small_font.render(name, True, (200, 200, 200))
            self.screen.blit(label, (bar_x, y - 5))
            
            # Background bar
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, y + 18, bar_width, bar_height))
            
            # Value bar
            value_width = int(bar_width * value)
            pygame.draw.rect(self.screen, color, (bar_x, y + 18, value_width, bar_height))
            
            # Border
            pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, y + 18, bar_width, bar_height), 2)
        
        # Instructions
        instructions = [
            "ESC: Quit",
            "SPACE: Restart",
        ]
        
        inst_y = self.height - 80
        for inst in instructions:
            inst_text = small_font.render(inst, True, (150, 150, 150))
            self.screen.blit(inst_text, (20, inst_y))
            inst_y += 25
    
    def run(self):
        """Main visualization loop"""
        self.running = True
        self.start_playback()
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS, dt in seconds
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.start_playback()
            
            # Update features based on time
            current_time = time.time()
            self.update_features_from_time(current_time)
            
            # Update particles
            self.update_particles(dt)
            
            # Draw
            self.draw_background()
            
            # Draw central circle (state indicator)
            center_color = self.stress_colors[0] if self.label == 'stress' else self.calm_colors[0]
            pygame.draw.circle(
                self.screen,
                center_color,
                (self.width // 2, self.height // 2),
                int(20 + 20 * self.alpha_power),
                3
            )
            
            # Draw particles
            for particle in self.particles:
                particle.draw(self.screen)
            
            # Draw info overlay
            self.draw_info()
            
            # Update display
            pygame.display.flip()
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 3: SWIM-inspired multimodal feedback visualizer"
    )
    parser.add_argument(
        '--features',
        required=True,
        help='Path to features CSV (e.g., outputs/demo/features_windows.csv)'
    )
    parser.add_argument(
        '--audio',
        required=True,
        help='Path to sonification WAV (e.g., outputs/demo/sonification.wav)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1200,
        help='Window width (default: 1200)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=800,
        help='Window height (default: 800)'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.features):
        print(f"Error: Features file not found: {args.features}")
        return
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Create visualizer
    print("Initializing SWIM visualizer...")
    viz = SWIMVisualizer(width=args.width, height=args.height)
    
    # Load data
    print(f"Loading features from: {args.features}")
    viz.load_features(args.features)
    
    print(f"Loading audio from: {args.audio}")
    viz.load_audio(args.audio)
    
    # Run
    print("Starting visualization...")
    print("\nControls:")
    print("  ESC: Quit")
    print("  SPACE: Restart playback")
    print()
    viz.run()


if __name__ == "__main__":
    main()