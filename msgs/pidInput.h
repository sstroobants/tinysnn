struct __attribute__((__packed__)) serial_pid_in {
    // thrust command (now used for resetting the pid)
    float thrust;
    //state
    float roll; //estimated roll
    float pitch; //estimated pitch
    float yaw; //estimated yaw
    //targets 
    float roll_t; //roll target
    float pitch_t; //pitch target
    float yaw_t; //yaw target

    //CHECKSUM
    uint8_t checksum_in;
};