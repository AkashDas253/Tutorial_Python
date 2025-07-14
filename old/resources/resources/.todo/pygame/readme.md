# Pygame Cheatsheet

## 1. Installing Pygame
- pip install pygame  # Install Pygame

## 2. Importing Pygame
- import pygame  # Import Pygame

## 3. Initializing Pygame
- pygame.init()  # Initialize all Pygame modules

## 4. Creating a Game Window
- screen = pygame.display.set_mode((width, height))  # Create game window
- pygame.display.set_caption('Game Title')  # Set window title

## 5. Main Game Loop
- running = True
- while running:
  - for event in pygame.event.get():  # Event loop
    - if event.type == pygame.QUIT:  # Check for quit event
      - running = False

## 6. Filling the Screen
- screen.fill((R, G, B))  # Fill screen with color (RGB)

## 7. Drawing Shapes
- pygame.draw.rect(screen, (R, G, B), (x, y, width, height))  # Draw rectangle
- pygame.draw.circle(screen, (R, G, B), (x, y), radius)  # Draw circle
- pygame.draw.line(screen, (R, G, B), (start_pos), (end_pos), width)  # Draw line

## 8. Loading Images
- image = pygame.image.load('image.png')  # Load image
- screen.blit(image, (x, y))  # Draw image on the screen

## 9. Handling Keyboard Input
- keys = pygame.key.get_pressed()  # Get current key states
- if keys[pygame.K_UP]:  # Check if UP key is pressed
  - # Do something

## 10. Updating the Display
- pygame.display.flip()  # Update the full display surface to the screen
- pygame.display.update()  # Update specific parts of the screen

## 11. Adding Sound
- pygame.mixer.init()  # Initialize the mixer
- sound = pygame.mixer.Sound('sound.wav')  # Load sound
- sound.play()  # Play sound

## 12. Controlling Frame Rate
- clock = pygame.time.Clock()  # Create a clock object
- clock.tick(60)  # Limit frame rate to 60 FPS

## 13. Quitting Pygame
- pygame.quit()  # Quit Pygame

