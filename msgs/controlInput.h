struct __attribute__((__packed__)) serial_control_in {
    //thrust (used for resetting network)
    float thrust;
    //state
    float roll; //roll target
    float pitch; //pitch target
    // gyro values
    float roll_gyro;
    float pitch_gyro;
    float yaw_gyro;
    // accelerometer values
    float x_acc;
    float y_acc;
    float z_acc;
    //CHECKSUM
    uint8_t checksum_in;
};