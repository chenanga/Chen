#include<iostream>
using namespace std;
#define MAX 1000 //最大人数
//通讯录管理系统
//系统中需要实现的功能如下：
//
//*添加联系人：向通讯录中添加新人，信息包括（姓名、性别、年龄、联系电话、家庭住址）最多记录1000人
//* 显示联系人：显示通讯录中所有联系人信息
//* 删除联系人：按照姓名进行删除指定联系人
//* 查找联系人：按照姓名查看指定联系人信息
//* 修改联系人：按照姓名重新修改指定联系人
//* 清空联系人：清空通讯录中所有信息
//* 退出通讯录：退出当前使用的通讯录


//封装函数显示主界面，在main函数中调用
void showMenu()
{
	cout << "**************************" << endl;
	cout << "*****  1、添加联系人  *****"  << endl;
	cout << "*****  2、显示联系人  *****" << endl;
	cout << "*****  3、删除联系人  *****" << endl;
	cout << "*****  4、查找联系人  *****" << endl;
	cout << "*****  5、修改联系人  *****" << endl;
	cout << "*****  6、清空联系人  *****" << endl;
	cout << "*****  0、退出通讯录  *****" << endl;
	cout << "**************************" << endl;
}


//联系人结构体
struct Person
{
	string m_Name; //姓名
	int m_Sex; //性别：1男 2女
	int m_Age; //年龄
	string m_Phone; //电话
	string m_Addr; //住址
};


//通讯录结构体
struct Addressbooks
{
	struct Person personArray[MAX]; //通讯录中保存的联系人数组
	int m_Size; //当前通讯录中人员个数
};


//1、添加联系人函数
void addPerson(struct Addressbooks* abs)
{
	if (abs->m_Size == MAX)
	{
		cout << "通讯录已满，无法添加。" << endl;
		return;
	}
	else
	{
		//添加联系人 姓名、性别、年龄、联系电话、家庭住址
		
		cout << "请输入待添加联系人的姓名" << endl;
		cin >> abs->personArray[abs->m_Size].m_Name ;

		cout << "请输入待添加联系人的性别" << endl;
		cout << "1 --- 男" << endl;
		cout << "2 --- 女" << endl;
		
		int sex;
		while (true)
		{

			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				abs->personArray[abs->m_Size].m_Sex = sex;
				break;
			}
			else
			{
				cout << "输入有误，请重新输入待添加联系人的性别" << endl;
			}
		}
		
		

		cout << "请输入待添加联系人的年龄" << endl;
		int age;
		
		while (true)
		{

			cin >> age;
			if (age >= 0 && age < 150)
			{
				abs->personArray[abs->m_Size].m_Age = age;
				break;
			}
			else
			{
				cout << "输入有误，请重新输入待添加联系人的年龄" << endl;
			}
		}


		cout << "请输入待添加联系人的联系电话" << endl;
		cin >> abs->personArray[abs->m_Size].m_Phone ;

		cout << "请输入待添加联系人的家庭住址" << endl;
		string address;
		cin >> abs->personArray[abs->m_Size].m_Addr ;

		abs->m_Size++;

		cout << "添加成功！" << endl;
		system("pause");  //请按任意键继续
		system("cls");    // 清屏操作
	}
}

//2、显示所有的联系人
void showperson(Addressbooks *abs)
{
	//判断是否为0 ，如果为0  提示记录空

	if (abs->m_Size == 0)
	{
		cout << "当前记录为空" << endl;
	}
	else
	{
		for (int i = 0; i < abs->m_Size; i++)
		{
			cout << "姓名： " << abs->personArray[i].m_Name << "\t";
			cout << "性别： " << (abs->personArray[i].m_Sex==1?"男":"女") << "\t";
			cout << "年龄： " << abs->personArray[i].m_Age << "\t";
			cout << "电话： " << abs->personArray[i].m_Phone << "\t";
			cout << "地址： " << abs->personArray[i].m_Addr << endl;
		}
	}
	system("pause");  //请按任意键继续
	system("cls");    // 清屏操作
}

//检测联系人是否存在，如果存在返回数组中的具体位置，不存在返回-1
int isExist(Addressbooks *abs,string name)
{
	for (int i = 0; i < abs->m_Size; i++)
	{
		if (abs->personArray[i].m_Name == name)
		{
			return i;
		}
		
	}
	return -1;
}

//3、删除联系人
void deletPerson(Addressbooks *abs)
{
	cout << "请输入删除联系人的姓名： " << endl;
	string name;
	cin >> name;
	int ret = isExist(abs, name);
	if (ret == -1)
		cout << "查无此人" << endl;
	else
	{
		for (int i = ret; i < abs->m_Size; i++)
		{
			//数据迁移
			abs->personArray[i] = abs->personArray[i + 1];
		}
		abs->m_Size--;
		cout << "删除成功" << endl;
	}
	system("pause");  //请按任意键继续
	system("cls");    // 清屏操作
}

//4、查找联系人
void findPerson(Addressbooks *abs)
{
	cout << "请输入要查找的联系人姓名：" << endl;
	string name;
	cin >> name;
	int ret = isExist(abs, name);
	if (ret != -1)
	{
		cout << "姓名： " << abs->personArray[ret].m_Name << "\t";
		cout << "性别： " << (abs->personArray[ret].m_Sex == 1 ? "男" : "女") << "\t";
		cout << "年龄： " << abs->personArray[ret].m_Age << "\t";
		cout << "电话： " << abs->personArray[ret].m_Phone << "\t";
		cout << "地址： " << abs->personArray[ret].m_Addr << endl;
	}
	else
	{
		cout << "查无此人" << endl;
	}
	
	system("pause");  //请按任意键继续
	system("cls");    // 清屏操作
}

//5、修改联系人
void modifyPerson(Addressbooks* abs)
{
	cout << "请输入您要修改的联系人" << endl;
	string name;
	cin >> name;
	int ret = isExist(abs, name);
	if (ret != -1)
	{
		//姓名
		cout << "请输入待修改联系人的姓名" << endl;
		cin >> abs->personArray[ret].m_Name;

		cout << "请输入待修改联系人的性别" << endl;
		cout << "1 --- 男" << endl;
		cout << "2 --- 女" << endl;

		int sex;
		while (true)
		{

			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				abs->personArray[ret].m_Sex = sex;
				break;
			}
			else
			{
				cout << "输入有误，请重新输入待修改联系人的性别" << endl;
			}
		}



		cout << "请输入待修改联系人的年龄" << endl;
		int age;

		while (true)
		{

			cin >> age;
			if (age >= 0 && age < 150)
			{
				abs->personArray[ret].m_Age = age;
				break;
			}
			else
			{
				cout << "输入有误，请重新输入待修改联系人的年龄" << endl;
			}
		}


		cout << "请输入待修改联系人的联系电话" << endl;
		cin >> abs->personArray[ret].m_Phone;

		cout << "请输入待修改联系人的家庭住址" << endl;
		string address;
		cin >> abs->personArray[ret].m_Addr;

		abs->m_Size++;

		cout << "修改成功！" << endl;
	}
	else
	{
		cout << "查无此人" << endl;
	}
	system("pause");  //请按任意键继续
	system("cls");    // 清屏操作
}

//6、清空所有联系人
void cleanPerson(Addressbooks* abs)
{
	abs->m_Size = 0;
	cout << "通讯录已清空" << endl;
	system("pause");  //请按任意键继续
	system("cls");    // 清屏操作
}

int main()
{
	//创建一个通讯录结构变量
	Addressbooks abs;
	abs.m_Size = 0;


	int select=0;  //创建用户选择输入的变量

	while (true)
	{
		//菜单调用
		showMenu();
		cin >> select;

		switch (select)
		{
		case 1:		//1、添加联系人
			addPerson(&abs);  //地址传递,可以修改实参
			break;
		case 2:		//2、显示联系人
			showperson(&abs);
			break;
		case 3:		//3、删除联系人
			deletPerson(&abs);
			break;
		case 4:		//4、查找联系人
			findPerson(&abs);
			break;
		case 5:		//5、修改联系人
			modifyPerson(&abs);
			break;
		case 6:		//6、清空联系人
			cleanPerson(&abs);
			break;
		case 0:		//7、退出通讯录
			cout << "欢迎下次使用" << endl;
			system("pause");
			return 0;
			break;
		default:
			break;


		}
	
	}
	
	system("pause");
	return 0;
}
