#include<stdlib.h>
#include<stdio.h>
#include<string>
#include"common/common.h"

size_t ReadFileToMem(std::string file_name, void* mem, int size){
    FILE *fp;
    fp=fopen(file_name.c_str(),"rb");
    if(!fp){
        LOG(ERROR)<<file_name<<" open failed";
        return -1;
    }
    auto rd=fread(mem,size,1,fp);
    /*
     * buffer 指向要读取的数组中首个对象的指针
size 每个对象的大小（单位是字节）
count 要读取的对象个数
stream 输入流
     * */
    fclose(fp);
    return rd;
}


size_t WriteMemToFile(std::string file_name, void* mem,int size){
    FILE*fp;
    fp = fopen(file_name.c_str(),"wb+");
    if(!fp){
        LOG(ERROR)<<file_name<<" open failed";
        return -1;
    }
    int rd=fwrite(mem, size,1, fp);
    fclose(fp);
    return rd;
}
