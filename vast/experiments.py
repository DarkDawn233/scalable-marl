import vast.rendering as rendering
from os.path import join
import vast.data as data
import numpy

def run_episode(env, controller, params, renderer, training_mode=True):
    state = env.global_state()
    done = False
    time_step = 0
    observations = env.reset()
    agent_infos = env.get_agent_infos()
    if training_mode:
        subteam_indices = controller.group_agents(agent_infos, time_step)
    sampled_subteams = None
    max_subteams = None
    while not done:
        joint_action = controller.policy(observations, time_step, training_mode)
        next_observations, rewards, done, _ = env.step(joint_action)
        next_state = env.global_state()
        time_step += 1
        if training_mode:
            result = controller.update(state, observations,\
                joint_action, rewards, next_state, next_observations, done, subteam_indices)
            subteam_indices = controller.group_agents(agent_infos, time_step)
            sampled_subteams, max_subteams = result["centralized_update"]
        agent_infos = env.get_agent_infos()
        state = next_state
        observations = next_observations
        renderer.render(env)
    if "directory" in params:
        data.save_json(join(params["directory"], "episode_0.json"), {"nothing":None})
    return_result = {
        "discounted_returns": env.discounted_returns.detach().numpy(),
        "undiscounted_returns": env.undiscounted_returns.detach().numpy(),
        "domain_value": env.domain_value(),
        "sampled_subteams": sampled_subteams,
        "max_subteams": max_subteams,
        "time_step": time_step
    }
    if hasattr(env, "win_rate"):
        return_result.update({"win_rate": env.win_rate()})
    return return_result

def run_episodes(nr_episodes, env, controller, params, training_mode=True):
    discounted_returns = [[] for _ in range(env.nr_agents)]
    undiscounted_returns = [[] for _ in range(env.nr_agents)]
    domain_values = []
    win_rates = []
    sampled_subteams = []
    max_subteams = []
    renderer = rendering.Renderer(params)
    time_step = 0
    for _ in range(nr_episodes):
        result = run_episode(env, controller, params, renderer, training_mode)
        if result["max_subteams"] is not None:
            max_subteams = result["max_subteams"]
        if result["sampled_subteams"] is not None:
            sampled_subteams = result["sampled_subteams"]
        for return_list, new_return in zip(discounted_returns, result["discounted_returns"]):
            return_list.append(new_return)
        for return_list, new_return in zip(undiscounted_returns, result["undiscounted_returns"]):
            return_list.append(new_return)
        domain_values.append(result["domain_value"])
        if result["win_rate"] is not None:
            win_rates.append(result["win_rate"])
        time_step += result["time_step"]
    renderer.close()
    return_result = {
        "discounted_returns": discounted_returns,
        "undiscounted_returns": undiscounted_returns,
        "domain_values": domain_values,
        "sampled_subteams": sampled_subteams,
        "max_subteams": max_subteams,
        "time_step": time_step
    }
    if len(win_rates) != 0:
        return_result.update({"win_rates":win_rates})
    return return_result

def run_training(env, controller, params):
    episodes_per_epoch = params["episodes_per_epoch"]
    evaluations_per_epoch = params["evaluations_per_epoch"]
    discounted_returns = [[]]
    undiscounted_returns = [[]]
    domain_values = []
    win_rates = []
    sampled_subteams = []
    max_subteams = []
    time_steps = []
    time_step = 0
    last_log = 0
    # for i in range(params["nr_epochs"]):
    while time_step <= params["t_max"]:
        training_result = run_episodes(episodes_per_epoch, env, controller, params, training_mode=True)
        time_step += training_result["time_step"]
        if time_step - last_log >= params["test_interval"]:
            last_log = time_step
            sampled_subteams.append(training_result["sampled_subteams"])
            max_subteams.append(training_result["max_subteams"])
            print("Finished step {} ({}, {}, {} agents ({})):".format(time_step, params["algorithm_name"], params["domain_name"], params["nr_agents"], params["nr_subteams"]))
            time_steps.append(time_step)
            result = run_episodes(evaluations_per_epoch, env, controller, params, training_mode=False)
            print("- Discounted return:  ", numpy.mean(result["discounted_returns"]))
            print("- Undiscounted return:", numpy.mean(result["undiscounted_returns"]))
            domain_value = float(numpy.mean(result["domain_values"]))
            domain_values.append(domain_value)
            if "win_rates" in result:
                win_rate = float(numpy.mean(result["win_rates"]))
                win_rates.append(win_rate)
                print("- Win rate: ", win_rate)
            print("- Domain value:", domain_value)
            for return_list, new_returns in zip(discounted_returns, result["discounted_returns"]):
                return_list.append(float(numpy.mean(new_returns)))
            for return_list, new_returns in zip(undiscounted_returns, result["undiscounted_returns"]):
                return_list.append(float(numpy.mean(new_returns)))
    result = {
        "discounted_returns": discounted_returns,
        "undiscounted_returns": undiscounted_returns,
        "domain_values": domain_values,
        # "sampled_subteams": sampled_subteams,
        # "max_subteams": max_subteams
    }
    if len(win_rates) != 0:
        result.update({"win_rates":win_rates})
    if "directory" in params:
        data.save_json(join(params["directory"], "results.json"), result)
        controller.save_model_weights(params["directory"])
    return result
