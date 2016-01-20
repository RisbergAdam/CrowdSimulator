#ifndef _tick_cuda_h_
#define _tick_cuda_h_

#include <iostream>
/**
 * These functions are placeholders for your functions 
 * implementing CUDA solutions. 
 * They are weakly linked. If there is any other function with the same signature at
 * link-time the conflict is resolved by scrapping the weak one.
 *
 * You should only care if you want to compile on a non-CUDA machine.
 */

void __attribute__((weak)) whereToGoCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}





#endif
