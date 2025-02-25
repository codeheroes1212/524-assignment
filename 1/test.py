import argparse
import torch
import cv2
import time
from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.avi")  
    args = parser.parse_args()
    return args

def test(opt):
    print("="*40)
    print("Test parameter configuration:")
    print(f"Game window width: {opt.width}")
    print(f"Game window height: {opt.height}")
    print(f"Square size: {opt.block_size}")
    print(f"Video FPS: {opt.fps}")
    print(f"Model loading path: {opt.saved_path}")
    print(f"Video output path: {opt.output}")
    print("="*40 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computing device detected: {device.upper()}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    print(f"in the process of being withdrawn from {opt.saved_path} Loading Models...")
    try:
        model = torch.load(f"{opt.saved_path}/tetris", map_location=torch.device(device))
        model.eval()
        print("Model loaded successfully！\n")
    except Exception as e:
        print(f"Model loading failure: {str(e)}")
        return

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    model.to(device)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(opt.output, fourcc, opt.fps,
                         (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    
    start_time = time.time()
    total_steps = 0
    print("Game Start！")
    print("-"*55)
    
    try:
        while True:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)
            
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            
            reward, done = env.step(action, render=True, video=out)
            total_steps += 1
            
            
            print(f"Step: {total_steps:04d} | " 
                  f"Action: {str(action):8} | "
                  f"Score: {env.score:5} | "
                  f"Cleared lines: {env.cleared_lines:3}")
            if done:
                end_time = time.time()
                print("\n" + "="*55)
                print("Game over! Final Statistics：")
                print(f"total number of steps: {total_steps}")
                print(f"Final score: {env.score}")
                print(f"Eliminate the total number of rows: {env.cleared_lines}")
                print(f"duration: {end_time - start_time:.2f} 秒")
                print(f"The video has been saved to: {opt.output}")
                print("="*55)
                break
                
    except KeyboardInterrupt:
        print("\nUser interrupted! Current progress is being saved...")
    finally:
        out.release()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
