#ifndef RL_HPP
#define RL_HPP

#include "agent.hpp"
#include "observation.hpp"
#include "env.hpp"

#include <ndn-cxx/util/scheduler.hpp>

namespace nlsr {

class RL {
public:
    RL(ConfParameter& ConfParameter, ndn::Scheduler& scheduler, vector<int> actions={0, 1})
    :m_env(ConfParameter, scheduler), m_agent(actions), m_scheduler(scheduler)
    {
    }

private:
    Env m_env;
    Agent m_agent;
    ndn::Scheduler& m_scheduler;
};

}

#endif