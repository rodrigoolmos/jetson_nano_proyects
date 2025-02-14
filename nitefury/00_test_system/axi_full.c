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

    if (pwrite(fd, &data[i], size, addr + i*4) == -1) {
        perror("pwrite");
        return -1;
    }
    return 0;
}

int read_bram(int fd, uint32_t addr, uint32_t *data, uint32_t size) {

    int i;

    if (pread(fd, &data[i], size, addr + i*4) == -1) {
        perror("pread");
        return -1;
    }
    return 0;
}

int blink_leds(int fd, uint32_t addr){

    int i;
    uint32_t led_output = 0;
    uint32_t leds_patron[100] = {
        // Segmento 1: Efecto "onda" (encendido progresivo y regresivo) (8 iteraciones)
        0x1, 0x3, 0x7, 0xF, 0x7, 0x3, 0x1, 0x0,
        
        // Segmento 2: Luz corrida (hacia adelante) (16 iteraciones)
        0x1, 0x2, 0x4, 0x8, 0x1, 0x2, 0x4, 0x8,
        0x1, 0x2, 0x4, 0x8, 0x1, 0x2, 0x4, 0x8,
        
        // Segmento 3: Parpadeo (todos encendidos y apagados) (10 iteraciones)
        0xF, 0x0, 0xF, 0x0, 0xF, 0x0, 0xF, 0x0, 0xF, 0x0,
        
        // Segmento 4: Patrón alterno (0xA y 0x5) (12 iteraciones)
        0xA, 0x5, 0xA, 0x5, 0xA, 0x5, 0xA, 0x5, 0xA, 0x5, 0xA, 0x5,
        
        // Segmento 5: Contador ascendente (de 0x0 a 0xF) (16 iteraciones)
        0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 
        0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF,
        
        // Segmento 6: Contador descendente (de 0xF a 0x0) (16 iteraciones)
        0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 
        0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
        
        // Segmento 7: Luz corrida (hacia atrás) (22 iteraciones)
        0x8, 0x4, 0x2, 0x1,   // ciclo 1 (4 iteraciones)
        0x8, 0x4, 0x2, 0x1,   // ciclo 2 (4 iteraciones)
        0x8, 0x4, 0x2, 0x1,   // ciclo 3 (4 iteraciones)
        0x8, 0x4, 0x2, 0x1,   // ciclo 4 (4 iteraciones)
        0x8, 0x4, 0x2, 0x1,   // ciclo 5 (4 iteraciones; 20 en total)
        0x8, 0x4            // 2 iteraciones extra para completar 22
    };
    
    

    // set as output
    if (pwrite(fd, &led_output, 4, addr + 4) == -1) {
        perror("pwrite");
        return -1;
    }

    for (i = 0; i < 100; i++) {
        if (pwrite(fd, &leds_patron[i], 4, addr) == -1) {
            perror("pwrite");
            return -1;
        }
        usleep(500000);
    }
    return 0;
}

int main(){

    int fd_h2c = open("/dev/xdma0_h2c_0", O_RDWR);
    if (fd_h2c == -1) {
        perror("open /dev/xdma0_h2c_0");
        return EXIT_FAILURE;
    }

    int fd_c2h = open("/dev/xdma0_c2h_0", O_RDWR);
    if (fd_c2h == -1) {
        perror("open /dev/xdma0_c2h_0");
        close(fd_h2c);
        return EXIT_FAILURE;
    }
    
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

    status = write_bram(fd_h2c, REG_ADDR, data_write, n_elem * sizeof(int));

    if (status == -1) {
        printf("Error writing to register\n");
        return -1;
    }

    status = read_bram(fd_c2h, REG_ADDR, data_read, n_elem * sizeof(int));

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
    printf("Test axi full passed\n");

    ////////////////////////////////////////////////////////////////
    /*                          BLINK LEDS                        */
    ////////////////////////////////////////////////////////////////

    blink_leds(fd_user, LEDS_ADDR);

    close(fd_user);
    free(data_write);
    free(data_read);
    return 0; 
}