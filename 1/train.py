import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque
import math
import pygame
from datetime import datetime

# PyGame initialisation
pygame.init()
UI_WIDTH, UI_HEIGHT = 800, 600
ui_screen = pygame.display.set_mode((UI_WIDTH, UI_HEIGHT))
pygame.display.set_caption("Tetris Training Monitor")

# Font initialisation (with alternatives)
try:
    title_font = pygame.font.SysFont('consolas', 24, bold=True)
    font = pygame.font.SysFont('consolas', 18)
except:
    title_font = pygame.font.Font(None, 24)
    font = pygame.font.Font(None, 18)

color_white = (255, 255, 255)
color_green = (50, 205, 50)
color_red = (220, 20, 60)
color_blue = (30, 144, 255)
color_gray = (169, 169, 169)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=0.1)
    parser.add_argument("--target_update", type=int, default=1000)
    parser.add_argument("--ui_refresh", type=int, default=1)

    args = parser.parse_args()
    return args

def draw_ui(epoch, opt, final_score, final_tetrominoes, final_cleared_lines,
            loss, epsilon, lr, q_values, duration, action, replay_memory):
    ui_screen.fill((0, 0, 40))  
    
    # Title area
    title_text = f"Tetris Training Monitor - Epoch {epoch}/{opt.num_epochs}"
    title_surf = title_font.render(title_text, True, color_white)
    ui_screen.blit(title_surf, (20, 15))
    
    # Key indicator regions
    y_pos = 70
    metrics = [
        (f"Score:", f"{final_score}", color_green),
        (f"Blocks:", f"{final_tetrominoes}", color_blue),
        (f"Lines:", f"{final_cleared_lines}", color_red),
        (f"Loss:", f"{loss:.4f}", color_white),
        (f"Epsilon:", f"{epsilon:.3f}", color_gray),
        (f"Learning Rate:", f"{lr:.2e}", color_gray),
        (f"Memory:", f"{len(replay_memory)}/{opt.replay_memory_size}", color_gray),
        (f"Duration:", f"{duration:.1f}s", color_gray),
        (f"Last Action:", f"{action}", color_blue)
    ]
    
    # Plotting left-hand side indicators
    for label, value, color in metrics:
        label_surf = font.render(label, True, color_white)
        value_surf = font.render(value, True, color)
        ui_screen.blit(label_surf, (30, y_pos))
        ui_screen.blit(value_surf, (200, y_pos))
        y_pos += 30
    
    # Q-value statistical regions
    y_pos = 70
    q_stats = [
        ("Q Min:", f"{torch.min(q_values):.2f}", color_green),
        ("Q Mean:", f"{torch.mean(q_values):.2f}", color_blue),
        ("Q Max:", f"{torch.max(q_values):.2f}", color_red),
        ("Q Std:", f"{torch.std(q_values):.2f}", color_white),
        ("Gamma:", f"{opt.gamma}", color_gray),
        ("Grad Clip:", f"{opt.grad_clip}", color_gray),
        ("Batch Size:", f"{opt.batch_size}", color_gray)
    ]
    
    # Plotting right-hand side indicators
    for label, value, color in q_stats:
        label_surf = font.render(label, True, color_white)
        value_surf = font.render(value, True, color)
        ui_screen.blit(label_surf, (400, y_pos))
        ui_screen.blit(value_surf, (580, y_pos))
        y_pos += 30
    
    # Score trend graphs
    if not hasattr(draw_ui, 'score_history'):
        draw_ui.score_history = []
    draw_ui.score_history.append(final_score)
    if len(draw_ui.score_history) > 50:
        draw_ui.score_history.pop(0)
    
    chart_x, chart_y = 30, 350
    chart_width, chart_height = 740, 200
    max_score = max(draw_ui.score_history) if draw_ui.score_history else 1
    
    # Drawing gridlines
    for i in range(0, 11):
        y = chart_y + chart_height - (i/10)*chart_height
        pygame.draw.line(ui_screen, (50, 50, 100), 
                       (chart_x, y), (chart_x+chart_width, y), 1)
        
    # Drawing trend lines
    if len(draw_ui.score_history) > 1:
        for i in range(1, len(draw_ui.score_history)):
            x1 = chart_x + (i-1)*(chart_width/(len(draw_ui.score_history)-1))
            y1 = chart_y + chart_height - (draw_ui.score_history[i-1]/max_score)*chart_height
            x2 = chart_x + i*(chart_width/(len(draw_ui.score_history)-1))
            y2 = chart_y + chart_height - (draw_ui.score_history[i]/max_score)*chart_height
            pygame.draw.line(ui_screen, color_green, (x1, y1), (x2, y2), 2)
    
    # Border
    pygame.draw.rect(ui_screen, color_white, 
                    (chart_x-2, chart_y-2, chart_width+4, chart_height+4), 2)
    
    # Scale labels
    if max_score > 0:
        max_label = font.render(f"Max: {max_score}", True, color_white)
        ui_screen.blit(max_label, (chart_x+10, chart_y-30))
    
    pygame.display.flip()

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    
   # Initialise the model
    model = DeepQNetwork()
    target_model = DeepQNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    criterion = nn.SmoothL1Loss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    
    # Initialise all display variables
    epoch = 0
    final_score = 0
    final_tetrominoes = 0
    final_cleared_lines = 0
    loss = torch.tensor(0.0)
    epsilon = opt.initial_epsilon
    action = (0, 0)
    best_score = -math.inf
    patience = 0
    start_time = datetime.now()
    
    # Initial UI rendering
    duration = (datetime.now() - start_time).total_seconds()
    draw_ui(
        epoch=0,
        opt=opt,
        final_score=final_score,
        final_tetrominoes=final_tetrominoes,
        final_cleared_lines=final_cleared_lines,
        loss=loss.item(),
        epsilon=epsilon,
        lr=opt.lr,
        q_values=torch.tensor([0.0]),
        duration=duration,
        action=action,
        replay_memory=replay_memory
    )
    
    # Train the main loop
    while epoch < opt.num_epochs:
        # Handling PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
        pygame.event.pump()
        
        next_steps = env.get_next_states()
        
        # ε decay strategy
        progress = min(epoch / opt.num_decay_epochs, 1.0)
        epsilon = opt.final_epsilon + (opt.initial_epsilon - opt.final_epsilon) * math.exp(-4 * progress)
        
        # Explore decision-making
        if random() <= epsilon:
            index = randint(0, len(next_steps) - 1)
            action_type = "Random"
        else:
            next_states = torch.stack(list(next_steps.values()))
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action_type = "AI"
        
        action = list(next_steps.keys())[index]
        next_state = list(next_steps.values())[index]
        reward, done = env.step(action, render=True)
        
        # Real-time update of current status
        current_score = env.score
        current_tetrominoes = env.tetrominoes
        current_cleared_lines = env.cleared_lines
        
        # Dynamic reward adjustments
        if done:
            reward -= 0.5  # Game over penalties
            # Record final results
            final_score = current_score
            final_tetrominoes = current_tetrominoes
            final_cleared_lines = current_cleared_lines
        else:
            reward += 0.1 * current_cleared_lines  # Eliminate row incentives
            # Continuous update of current status
            final_score = current_score
            final_tetrominoes = current_tetrominoes
            final_cleared_lines = current_cleared_lines
            
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        
        replay_memory.append([state, reward, next_state, done])
        state = next_state if not done else env.reset()
        
         # Early stop mechanism
        if done:
            if final_score > best_score:
                best_score = final_score
                patience = 0
                torch.save(model.state_dict(), f"{opt.saved_path}/best_model.pth")
            else:
                patience += 1
                if patience > 200 and epoch > 500:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if torch.cuda.is_available():
                state = state.cuda()
        
        if len(replay_memory) < opt.replay_memory_size // 10:
            continue
        
        # Start training
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.stack(state_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch)
        
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        
        # Calculate target Q
        with torch.no_grad():
            target_q = target_model(next_state_batch)
        max_target_q = target_q.max(1)[0]
        y_batch = reward_batch + (1 - torch.FloatTensor(done_batch)) * opt.gamma * max_target_q
        
        # Training steps
        optimizer.zero_grad()
        q_values = model(state_batch).squeeze()
        loss = criterion(q_values, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        scheduler.step()
        
        # Update the target network
        if epoch % opt.target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
       # Real-time UI updates
        duration = (datetime.now() - start_time).total_seconds()
        draw_ui(
            epoch=epoch,
            opt=opt,
            final_score=final_score,
            final_tetrominoes=final_tetrominoes,
            final_cleared_lines=final_cleared_lines,
            loss=loss.item(),
            epsilon=epsilon,
            lr=optimizer.param_groups[0]['lr'],
            q_values=q_values,
            duration=duration,
            action=action,
            replay_memory=replay_memory
        )
        
        # Logging
        writer.add_scalar('Train/Score', final_score, epoch)
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/Epsilon', epsilon, epoch)
        writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Console output
        if epoch % 50 == 0:
            print(f"Epoch: {epoch:5d} | Score: {final_score:6.0f} | "
                  f"Loss: {loss.item():6.3f} | ε: {epsilon:.3f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Q: {torch.mean(q_values):.2f}")
        
        # Save the model
        if epoch % opt.save_interval == 0:
            torch.save(model, f"{opt.saved_path}/tetris_{epoch}.pth")
    
    # End-of-training processing
    pygame.quit()
    torch.save(model, f"{opt.saved_path}/final_model.pth")
    print("Training completed.")

if __name__ == "__main__":
    opt = get_args()
    train(opt)





