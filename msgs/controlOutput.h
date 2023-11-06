struct __attribute__((__packed__)) serial_control_out {
    //torque commands
    float torque_x; //torque x
    float torque_y; //torque y
    float x_integ;
    float y_integ;
    //CHECKSUM
    uint8_t checksum_out;
};