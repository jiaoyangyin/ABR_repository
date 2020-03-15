#include <stdlib.h>
#include "Python.h"


PyObject* import_python_module(char *modulename) {
  if(NULL == modulename) return NULL;
  PyObject* pModule = PyImport_ImportModule(modulename);
  if(NULL == pModule ) {
    printf("Python get module(%s) failed.\n", modulename);
  }
  return pModule ;
}

int main(int argc, char* argv[])
{
  //初始化Python环境 
  Py_Initialize();
  PyRun_SimpleString("import sys");
  //添加Insert模块路径 
  //PyRun_SimpleString(chdir_cmd.c_str());
  PyRun_SimpleString("sys.path.append('./')");
  //导入模块 
  PyObject* pModule = import_python_module("hello");
  if (!pModule)
  {
    return 0;
  }
  printf("Python get module succeed.\n");

  //获取Insert模块内_add函数 
  PyObject* pv = PyObject_GetAttrString(pModule, "test");
  if (!pv || !PyCallable_Check(pv))
  {
    printf("Can't find function (test)\n");
    return 0;
  }
  printf("Get function (test) succeed.\n");
  //初始化要传入的参数，args配置成传入两个参数的模式 
  PyObject* args = PyTuple_New(1);
  //将Long型数据转换成Python可接收的类型 
  //PyObject* arg1 = PyLong_FromLong(4);
  //PyObject* arg2 = PyLong_FromLong(3);
  //将arg1配置为arg带入的第一个参数 
  //PyTuple_SetItem(args, 0, arg1);
  //将arg1配置为arg带入的第二个参数 
  //PyTuple_SetItem(args, 1, arg2);
  //传入参数调用函数，并获取返回值 

  /*
  char aaa[] = "aaa";
  PyObject * bytesObject = PyByteArray_FromStringAndSize(aaa, strlen(aaa));
  PyObject * buildObject = Py_BuildValue("y#", aaa,strlen(aaa));
  PyTuple_SetItem(args, 0, bytesObject);
  PyObject* pRet = PyObject_CallObject(pv, args);
  if (pRet)
  {
    //将返回值转换成long型 
    long result = PyLong_AsLong(pRet);
    printf("result: %ld\n", result);
  }
  */
  Py_Finalize();
  system("pause");
  return 0;
}
