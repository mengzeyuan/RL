#include "rl.hpp"

#define max_episodes 3

using namespace std;

namespace nlsr{

    void RL::startRL(double seconds)
    {
        std::cout<< "进入RL::startRL" <<std::endl;
        ns3::Simulator::Schedule(ns3::Seconds(seconds), &RL::update_1, this);
    }

    void RL::update_1()
    {
        std::cout<< "进入RL::update_1" <<std::endl;
        vector<double> current_state = m_env.return_state();

        update_2(current_state);

        //std::future<tuple<bool, double, Observation>> future = std::async(&Env::getRewardAndObservation, &m_env);
        //tuple<bool, double, Observation> result = future.get();
        //std::cout<< std::get<0>(result) <<std::endl;

        //如果写成callback的形式呢？
        //ns3::Callback<tuple<bool, double, nlsr::Observation>> my_cb;
        //my_cb = ns3::MakeCallback(&Env::getRewardAndObservation, &m_env);
    }

    void RL::update_2(const vector<double>& current_state)
    {
        std::cout<< "进入RL::update_2" <<std::endl;
        int action = m_agent.choose_action(current_state);
        m_env.step(action);
        ns3::Simulator::Schedule(ns3::Seconds(20.0), &RL::update_3, this, action, current_state);
    }

    void RL::update_3(const int& action, const vector<double>& current_state)
    {
        std::cout<< "进入RL::update_3" <<std::endl;
        double reward = m_env.get_reward();
        vector<double> next_state = m_env.return_state();
        cout<<"Reward: "<<reward<<endl;
        cout<<"Episodes: "<<m_env.episodes<<" Current episode frames: "<<m_env.current_episode_frames<<" Frames: "<<m_agent.frames<<endl;
        m_agent.store_mem(current_state, action, reward, next_state, m_env.get_is_done());
        if(m_agent.frames>300 && m_agent.frames%5==0) {
            m_agent.train();
        }
        update_2(next_state);
    }

}