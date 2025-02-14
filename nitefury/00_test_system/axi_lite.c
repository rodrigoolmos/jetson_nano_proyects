#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>

#define REG_ADDR 0x00000000
#define LEDS_ADDR 0x10000
#define REG_SIZE 0x10000

int write_bram(int fd, uint32_t addr, uint32_t *data, uint32_t size) {

    int i;

    for (i = 0; i < size; i++) {
        if (pwrite(fd, &data[i], 4, addr + i*4) == -1) {
            perror("pwrite");
            return -1;
        }
    }
    return 0;
}

int read_bram(int fd, uint32_t addr, uint32_t *data, uint32_t size) {

    int i;

    for (i = 0; i < size; i++) {
        if (pread(fd, &data[i], 4, addr + i*4) == -1) {
            perror("pread");
            return -1;
        }
    }
    return 0;
}

int blink_leds(int fd, uint32_t addr){

    int i;
    uint32_t led_output = 0;
    uint32_t leds_patron[20] = {
        0x0000000A, 0x00000005, 0x0000000A, 0x00000005, 0x000000A,
        0x00000001, 0x00000002, 0x00000003, 0x00000004, 0x00000005,
        0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000009,
        0x0000000F, 0x00000000, 0x0000000F, 0x00000000, 0x0000000F};

    // set as output
    if (pwrite(fd, &led_output, 4, addr + 4) == -1) {
        perror("pwrite");
        return -1;
    }

    for (i = 0; i < 20; i++) {
        if (pwrite(fd, &leds_patron[i], 4, addr) == -1) {
            perror("pwrite");
            return -1;
        }
        usleep(500000);
    }
    return 0;
}

int main(){

    
    int fd_user = open("/dev/xdma0_user", O_RDWR);
    if (fd_user < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////////////////////
    /*                           TEST BRAM                        */
    ////////////////////////////////////////////////////////////////
    
    int n_elem = REG_SIZE / sizeof(int);
    int *data_write = (int *)malloc(n_elem * sizeof(int));
    int *data_read = (int *)malloc(n_elem * sizeof(int));

    int i, status;

    for (i = 0; i < n_elem; i++) {
        data_write[i] = i;
    }

    status = write_bram(fd_user, REG_ADDR, data_write, n_elem);

    if (status == -1) {
        printf("Error writing to register\n");
        return -1;
    }

    status = read_bram(fd_user, REG_ADDR, data_read, n_elem);

    if (status == -1) {
        printf("Error reading from register\n");
        return -1;
    }

    for (i = 0; i < n_elem; i++) {
        printf("data_write %d \t data_read %d \n", data_write[i], data_read[i]);
        if(data_write[i] != data_read[i]) {
            printf("Error: data_write != data_read\n");
            return -1;
        }
    }
    printf("Test axi lite passed\n");

    ////////////////////////////////////////////////////////////////
    /*                          BLINK LEDS                        */
    ////////////////////////////////////////////////////////////////

    blink_leds(fd_user, LEDS_ADDR);

    close(fd_user);
    free(data_write);
    free(data_read);
    return 0; 
}