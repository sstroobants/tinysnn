

struct __attribute__((__packed__)) serial_pid_in {
    //state
    int16_t roll; //estimated roll
    int16_t pitch; //estimated pitch
    int16_t yaw; //estimated yaw
    //targets 
    int16_t roll_t; //roll target
    int16_t pitch_t; //pitch target
    int16_t yaw_t; //yaw target
     //Rolling message out
    float rolling_msg_in;
    uint8_t rolling_msg_in_id;
    //CHECKSUM
    uint8_t checksum_in;
};

struct __attribute__((__packed__)) serial_pid_out {
    //roll commands
    int16_t roll_p; //roll p
    int16_t roll_i; //roll i
    int16_t roll_d; //roll d
    //pitch commands
    int16_t pitch_p; //pitch p
    int16_t pitch_i; //pitch i
    int16_t pitch_d; //pitch d
    //yaw commands
    int16_t yaw_p; //yaw p
    int16_t yaw_i; //yaw i
    int16_t yaw_d; //yaw d
     //Rolling message out
    float rolling_msg_in;
    uint8_t rolling_msg_in_id;
    //CHECKSUM
    uint8_t checksum_in;
};