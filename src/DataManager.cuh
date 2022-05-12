#ifndef DATA_MANAGER_CUH
#define DATA_MANAGER_CUH

#include <sys/time.h>

/**
 * Class used to store times for a specific algorithm and generate statistics on
 * the set of time data.
 */
class DataManager{
  
  private:
    struct timeval startTime, endTime, elapsedTime;

  public:
    /**
     * @brief Simple constructor.
     */
    DataManager();
    
    /**
     * @brief Records the start time.
     * 
     */
    void startTimer();

    /**
     * @brief Records the end time.
     * 
     */
    void stopTimer();

    /**
     * @brief Gets the total time elapsed between start and stop of the timer.
     * 
     * @return double 
     */
    double getTimeElapsed();
    
};

#endif // DATA_MANAGER_CUH
