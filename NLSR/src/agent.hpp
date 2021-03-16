#ifndef AGENT_HPP
#define AGENT_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "NeuralNet.cpp"
using namespace std;

namespace nlsr{

/* class ReplayMemory {
private:
    vector<vector<vector<double>>> mem;
    int capacity;
public:
    vector<double> current_state;
    int action;
    int reward;
    vector<double> next_state;
    bool is_done;
    ReplayMemory () {}
    ReplayMemory (int capacity) {
        this->capacity = capacity;
        srand(time(NULL));
    }
    void store (vector<double> current_state,
		    int action,
		    int reward,
		    vector<double> next_state,
		    bool is_done) {
        if (this->mem.size() == capacity) {
            this->mem.erase(this->mem.begin());
        }

        vector<vector<double>> append_mem;
        append_mem.push_back(current_state);
        append_mem.push_back({(double)action});
        append_mem.push_back({(double)reward});
        append_mem.push_back(next_state);
        append_mem.push_back({(double)is_done});
        this->mem.push_back(append_mem);
    }

    void random () {
        int random_num = rand() % this->mem.size();
        vector<vector<double>> mem_read = this->mem[random_num];
        this->current_state = mem_read[0];
        this->action = (int)mem_read[1][0];
        this->reward = (int)mem_read[2][0];
        this->next_state = mem_read[3];
        this->is_done = (bool)mem_read[4][0];
    }
}; */

class ReplayMemory {
private:
    int capacity;
    std::ofstream outfile;
    string router_name;

public:
    vector<double> current_state;
    int action;
    int reward;
    vector<double> next_state;
    bool is_done;
    ReplayMemory () {}
    void initialize (int capacity, const string& routerName) {
        this->router_name = routerName;
        this->capacity = capacity;
        srand(time(NULL));
        //cout<<routerName.toUri()<<endl;   // /n/e/%C1r0
        this->outfile.open("node"+router_name.substr(9,router_name.size())+"_mem.txt");
        if (this->outfile.is_open())
        {std::cout <<"node"+router_name.substr(9,router_name.size())+"_mem.txt"<<"被创建" << std::endl;}
    }
    ~ReplayMemory () {
        this->outfile.close();
    }
    void store (vector<double> current_state,
        int action,
        int reward,
        vector<double> next_state,
        bool is_done) {
            //of_hello << Simulator::Now().ToDouble(Time::S) << "\t" << nodeName << "\t" << arg1 << "\t" << arg2 << "\t" << arg3 << "\t" << arg4 << "\t" << arg5 << "\t" << arg6 << endl; 
            this->outfile << current_state[0] << " " << current_state[1] << " " << action << " " << reward << " " << next_state[0] << " " << next_state[1] << " " << is_done << endl;
    }
    string read_line(string filename,int line)
    {
        int i=0;
        string temp;
        fstream file;
        file.open(filename,ios::in);
        if(line<=0)
        {
            return "Error 1: 行数错误，不能为0或负数";
        }
        if(file.fail())
        {
            return "Error 2: 文件不存在";
        }
        while(getline(file,temp)&&i<line-1)
        {
            i++;
        }
        file.close();
        return temp;
    }
    void random () {
        string temp = read_line("node"+router_name.substr(9,router_name.size())+"_mem.txt",1);
        cout<<temp<<endl;
        vector<string> fields;
        boost::split(fields, temp, boost::is_any_of(" "));
        current_state.clear();
        next_state.clear();
        this->current_state.push_back(std::stod(fields[0]));
        this->current_state.push_back(std::stod(fields[1]));
        this->action = std::stoi(fields[2]);
        this->reward = std::stoi(fields[3]);
        this->next_state.push_back(std::stod(fields[4]));
        this->next_state.push_back(std::stod(fields[5]));
        this->is_done = (bool)std::stoi(fields[6]);
    }
    
};

class Agent
{
public:
    int frames = 0;
    vector<double> last_prediction;
    Agent() {

    }
    Agent (vector<int> layout, double lr, int mem_capacity, int frameReachProb, int targetFreqUpdate, int batches)
    :frameReachProb(frameReachProb), targetFreqUpdate(targetFreqUpdate), batches(batches)
    {
        this->net = NeuralNetwork(layout, lr);
        this->target_net = net;
        srand(time(NULL));
    }
    //求最大值下标
    int argmax (vector<double> array);
    int choose_action (vector<double> input);
    void store_mem (vector<double> current_state, int action, int reward, vector<double> next_state, bool is_done);
    double max (vector<double> array);
    void train ();
    void initialize_mem (const string& router_name);

private:
    NeuralNetwork net = NeuralNetwork();
    NeuralNetwork target_net = NeuralNetwork();
    ReplayMemory mem;
    int frameReachProb;
    int batches;
    int targetFreqUpdate;
    //bool is_folder_created = 1;
};

}



#endif