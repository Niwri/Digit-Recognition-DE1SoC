#define RESOLUTION_X 320
#define RESOLUTION_Y 240

#define CENTER_X 160
#define CENTER_Y 120

#define CHAR_COL 80
#define CHAR_ROW 60

#define CENTER_COL 40
#define CENTER_ROW 30

#define SDRAM_BASE            0xC0000000
#define FPGA_ONCHIP_BASE      0xC8000000
#define FPGA_CHAR_BASE        0xC9000000

#define LEDR_BASE             0xFF200000
#define HEX3_HEX0_BASE        0xFF200020
#define HEX5_HEX4_BASE        0xFF200030
#define SW_BASE               0xFF200040
#define KEY_BASE              0xFF200050
#define TIMER_BASE            0xFF202000
#define PIXEL_BUF_CTRL_BASE   0xFF203020
#define CHAR_BUF_CTRL_BASE    0xFF203030

#define SCALESIZE 7
#define SIZE 28

#define BORDER_LEFT 16
#define BORDER_RIGHT 302

#define BORDER_TOP 16
#define BORDER_BOTTOM 222

/************************************************************************************
*                                                                                   *
*   Graphic Arrays                                                                  *
*                                                                                   *
*************************************************************************************/

// Background and Title
#include "graphicHeaders/titleBackground.h"
#include "graphicHeaders/title.h"

// Button template
#include "graphicHeaders/button.h"
#include "graphicHeaders/buttonClick.h"

// Start Text
#include "graphicHeaders/train.h"
#include "graphicHeaders/trainClick.h"

// Load Text
#include "graphicHeaders/load.h"

/************************************************************************************
*                                                                                   *
*   Enums                                                                           *
*                                                                                   *
*************************************************************************************/

typedef enum {
    START = 0,
    LOAD = 1,
    MENU = 2,
    CANVAS = 3
} Page;

typedef enum {
    true = 1,
    false = 0
} bool;

typedef enum {
    NONE = 0,
    HIGHLIGHT_TRAIN = 1,
    HIGHLIGHT_DRAW = 2,
    DRAW = 3,
} Handle;

typedef void (*page_draw_ptr)();
typedef void (*page_handle_ptr)();
typedef void (*handle_draw_ptr)();

typedef struct {
    int x;
    int y;
} Position;

typedef struct {
    int ySize;
    int xSize;
} Size;

/************************************************************************************
*                                                                                   *
*   Global Variables                                                                *
*                                                                                   *
*************************************************************************************/

int switchPageCount;
int handleNumber;
volatile int pixel_buffer_start;


/************************************************************************************
*                                                                                   *
*   Basic Drawing                                                                   *
*                                                                                   *
*************************************************************************************/

void write_char(int x, int y, char c) {
    // VGA character buffer
    volatile char * character_buffer = (char *) (FPGA_CHAR_BASE + (y<<7) + x);
    *character_buffer = c;
}

void plot_pixel(int x, int y, short int line_color) {
    *(short int *)(pixel_buffer_start + (y << 10) + (x << 1)) = line_color;
}

/************************************************************************************
*                                                                                   *
*   Render Functions                                                                *
*                                                                                   *
*************************************************************************************/

void drawComponent(Position pos, Size size, short int component[size.ySize][size.xSize]) {
    for(int y = 0; y < size.ySize; y++)
        for(int x = 0; x < size.xSize; x++)
            // Consider black pixels as transparent
            if(component[y][x] != 0x0)
                plot_pixel(pos.x + x, pos.y + y, component[y][x]);
}

void writeText(Position pos, char* text) {
    int x = pos.x;

    while(*text) {
        write_char(x, pos.y, *text);
        x++;
        text++;
    }
}

/************************************************************************************
*                                                                                   *
*   Page Rendering                                                                  *
*                                                                                   *
*************************************************************************************/

void startRender() {

    // Draw title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);
    
    // Draw button
    Size buttonSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position buttonPos = {CENTER_X - buttonSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - buttonSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(buttonPos, buttonSize, button);
    
    // Draw train text
    char* text = "TRAIN MODEL";
    Position textPos = {CENTER_COL - 5, CHAR_ROW * 2.0 / 3.0};
    writeText(textPos, text);
    
}

void loadRender() {
    // Draw title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);

    char* text = "LOADING...";
    Position textPos = {CHAR_COL / 2 - 5, CHAR_ROW * 2.0/3.0};
    writeText(textPos, text);
}

void menuRender() {
    // Draw title
    Size titleSize = {sizeof(title) / sizeof(title[0]), sizeof(title[0]) / sizeof(title[0][0])};
    Position titlePos = {CENTER_X - titleSize.xSize / 2, BORDER_TOP + 40};
    drawComponent(titlePos, titleSize, title);
    
    // Draw button
    Size drawSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position drawPos = {CENTER_X - drawSize.xSize/2, RESOLUTION_Y * 1.0 / 2.0 - drawSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(drawPos, drawSize, button);

    // Exit button
    Size exitSize = {sizeof(button) / sizeof(button[0]), sizeof(button[0]) / sizeof(button[0][0])};
    Position exitPos = {CENTER_X - exitSize.xSize/2, RESOLUTION_Y * 2.0 / 3.0 - exitSize.ySize/2 + RESOLUTION_Y / CHAR_ROW / 2};
    drawComponent(exitPos, exitSize, button);

    char* drawText = "DRAW NOW";
    Position drawTextPos = {CHAR_COL / 2 - 4, CHAR_ROW * 1.0/2.0};
    writeText(drawTextPos, drawText);

    char* exitText = "EXIT";
    Position exitTextPos = {CHAR_COL / 2 - 2, CHAR_ROW * 2.0/3.0};
    writeText(exitTextPos, exitText);
}

void canvasRender() {
        
}

page_draw_ptr drawPage[] = {startRender, loadRender, menuRender, canvasRender};


/************************************************************************************
*                                                                                   *
*   Page Handlers                                                                   *
*                                                                                   *
*************************************************************************************/

void startHandle() {
    /*
        No Handle = 0
        Train Button Hover = 1
        Train Button Click = 2
    */
}

void loadHandle() {
    
}

void menuHandle() {
    
}

void canvasHandle() {
    
}

page_handle_ptr handlePage[] = {startHandle, loadHandle, menuHandle, canvasHandle};

/************************************************************************************
*                                                                                   *
*   Handle Event Rendering                                                          *
*                                                                                   *
*************************************************************************************/

void noHandle() {
    return;
}

handle_draw_ptr handleRender[] = {noHandle};

/************************************************************************************
*                                                                                   *
*   Buffering                                                                       *
*                                                                                   *
*************************************************************************************/


void wait_for_vsync() {
	volatile int* buffer = PIXEL_BUF_CTRL_BASE;
    
	*buffer = 1;
	register int status = *(buffer+3);
	
	while((status & 0x01) != 0)
		status = *(buffer+3);
}

void clear_screen() {
	for(int x = BORDER_LEFT; x < BORDER_RIGHT; x++) 
		for(int y = BORDER_TOP; y < BORDER_BOTTOM; y++)
			plot_pixel(x, y, 0x0);	
}

void clear_character() {
    for(int x = 0; x < CHAR_COL; x++)
        for(int y = 0; y < CHAR_ROW; y++)
            write_char(x, y, ' ');
}

void drawBackground() {
    for(int i = 0; i < 240; i++)
        for(int j = 0; j < 320; j++)
            plot_pixel(j, i, titleBackground[i][j]);
}



void main() {

    /* Set up page */
    Page currentPage = MENU;
    switchPageCount = 0;
    handleNumber = 0;

    /* Set up buffers */
    volatile int * pixel_ctrl_ptr = (int*)PIXEL_BUF_CTRL_BASE;
    
    clear_character();

    *(pixel_ctrl_ptr + 1) = FPGA_ONCHIP_BASE; 
    wait_for_vsync();
    
    pixel_buffer_start = *pixel_ctrl_ptr;
    drawBackground();
    drawPage[currentPage]();

    *(pixel_ctrl_ptr + 1) = SDRAM_BASE;
    
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    drawBackground(); 
    drawPage[currentPage]();

    
	
    while (1)
    {

        // Handle user input via polling depending on page and changes handleNumber if event handle occured 
        handlePage[currentPage]();

        // Draw the next page on back buffer. Draws again when swapped buffers.
        if(switchPageCount < 2) {
            clear_screen();
            drawPage[currentPage]();
            switchPageCount++;
        }

        // Render any event handles that occured from handlePage
        handleRender[handleNumber]();
        
        wait_for_vsync(); // swap front and back buffers on VGA vertical sync
        pixel_buffer_start = *(pixel_ctrl_ptr + 1); // new back buffer
    }
}

