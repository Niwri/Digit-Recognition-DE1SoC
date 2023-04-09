
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

typedef enum {
    PENCIL,
    ERASE
} Mode;

typedef enum {
    WAIT_ACKNOWLEDGE,
    REPORTING,
} Status;

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


// Function declaration
void write_char(int x, int y, char c);
void plot_pixel(int x, int y, short int line_color);
bool mouseIsInside(int minX, int maxX, int minY, int maxY);
void drawComponent(Position pos, Size size, short int component[size.ySize][size.xSize]);
void writeText(Position pos, char* text);
void drawCursor(int mx, int my);
void clearCanvas();
void startRender();
void loadRender();
void menuRender();
void canvasRender();
void startHandle();
void loadHandle();
void menuHandle();
void canvasHandle();
void noHandle();
void trainButtonHover();
void trainButtonNoHover();
void trainButtonClick();
void loadModel();
void drawButtonHover();
void drawButtonNoHover();
void exitButtonHover();
void exitButtonNoHover();
void drawButtonClick();
void exitButtonClick();
void drawCanvasArray();
void backButtonHover();
void backButtonNoHover();
void recognizeButtonHover();
void recognizeButtonNoHover();
void backButtonClick();
void recognizeButtonClick();
void modeButtonClick();
void wait_for_vsync();
void clear_screen();
void clear_character();
void drawBackground();
void displayResult(int num);
void mouseInput();
void config_GIC(void);
void config_PS2(void);
void set_A9_IRQ_stack(void);
void enable_A9_interrupts(void);
void config_interrupt(int N, int CPU_target);
void __attribute__((interrupt)) __cs3_isr_irq(void);
void __attribute__((interrupt)) __cs3_reset(void);
void __attribute__((interrupt)) __cs3_isr_undef(void);
void __attribute__((interrupt)) __cs3_isr_swi(void);
void __attribute__((interrupt)) __cs3_isr_pabort(void);
