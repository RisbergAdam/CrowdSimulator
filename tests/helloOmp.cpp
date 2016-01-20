#include <omp.h>
#include <iostream>

  


int main()
{
  int target_thread_num = 2;
  omp_set_num_threads(target_thread_num);
  int check[target_thread_num] = {0};
  
// Initialize all the times
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    
    check[thread_id] = 1;
  }
  
  int count = 0;
  for(int i = 0; i < target_thread_num;i++)
  {
    count += check[i];
  }
  std:: cout << count << " Threads were spawned\n";
  
  return (count < target_thread_num);
}
