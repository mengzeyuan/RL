#include "env.hpp"

namespace nlsr {

Env::Env(ConfParameter& ConfParameter)
: m_confParameter(ConfParameter)
//, m_scheduler(scheduler)
{
    reset();
}

void Env::reset()
{
    m_confParameter.setInfoInterestInterval(defaultHelloInterval);
    m_confParameter.setRoutingCalcInterval(defaultRoutingCalcInterval);
    is_done = false;
    episodes++;
    current_episode_frames = 0;
    total_reward = 0;
}

void Env::step(const int &action)
{
    current_episode_frames++;
    uint32_t curHelloInterval = m_confParameter.getInfoInterestInterval();
    uint32_t curRoutingCalcInterval = m_confParameter.getRoutingCalcInterval();
    switch (action)
        {
        case 0: //increase HelloInterval
            if(curHelloInterval<88){
                m_confParameter.setInfoInterestInterval(curHelloInterval+2);
            }
        break;
        case 1: //decrease HelloInterval
            if(curHelloInterval>5){
                m_confParameter.setInfoInterestInterval(curHelloInterval-2);
            }
        break;
        case 2: //increase RoutingCalculateInterval
            if(curRoutingCalcInterval<20){
                m_confParameter.setRoutingCalcInterval(curRoutingCalcInterval+2);
            }
        break;
        case 3: //decrease RoutingCalculateInterval
            if(curRoutingCalcInterval>5){
                m_confParameter.setRoutingCalcInterval(curRoutingCalcInterval-2);
            }
        break;

        default:
            break;
        }
}

double Env::get_reward()
{
    double reward;
    is_done = false;
    interestHitRatio = (double)ns3::ndn::Consumer::numInData/ns3::ndn::Consumer::numOutInterests;
    interestTimeoutRatio = (double)ns3::ndn::Consumer::numTimeOutInterests/ns3::ndn::Consumer::numOutInterests;
    //std::cout<< "进入Env::getRewardAndObservation" <<std::endl;
    if(interestHitRatio > 0.9 && interestTimeoutRatio < 0.2) {
        reward = 5;
        is_done = true;
        reset();
    }
    else {
        double distance = abs(0.9-interestHitRatio) + abs(interestTimeoutRatio-0.2);
        reward = -distance;
    }
    total_reward += reward;
    return total_reward;
}

vector<double> Env::return_state() {
    interestHitRatio = (double)ns3::ndn::Consumer::numInData/ns3::ndn::Consumer::numOutInterests;
    interestTimeoutRatio = (double)ns3::ndn::Consumer::numTimeOutInterests/ns3::ndn::Consumer::numOutInterests;
    vector<double> state;
    state.push_back(interestHitRatio);
    state.push_back(interestTimeoutRatio);
    return state;
}

bool Env::get_is_done() {
    return is_done;
}

}