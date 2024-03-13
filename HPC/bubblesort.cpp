#include<iostream>
#include<stdlib.h>
#include<omp.h>
using namespace std;


int bubble_sort(int arr[], int n)
#pragma omp parallel
{
	int i, temp, flag=1;
	while(flag)
	{
		flag=0;
		#pragma omp parallel for
		for (int i=0; i<n-1;i++)
		{
			if(arr[i] > arr[i+1])
			{
				temp = arr[i];
				arr[i] = arr[i+1];
				arr[i+1] = temp;
				flag=1;
			}
		}
	}
}

int main()
{
	int arr[] = {5,2,8,9,4,1,6,7};
	int n = sizeof(arr) / sizeof(arr[0]);
	
	bubble_sort(arr[], n);
	
	cout<<"Sorted Array: ";
	
	for(int i=0;i<n;i++)
	{
		cout<<arr[i]<<"";
	}
	
	
	return 0;
}
