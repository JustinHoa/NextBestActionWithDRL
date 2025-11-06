from gymnasium.envs.registration import register

register(
    id="NextBestActionEnv-v0",                       
    entry_point="next_best_action_env.env:NextBestActionEnv",  
    max_episode_steps=50,                            
)
