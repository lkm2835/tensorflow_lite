#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {

	string fileName = "log.txt";

	if (argc >= 2) fileName = argv[1];
    
	ifstream myfile(fileName);
	string buffer;

	while (myfile.peek() != EOF) {
		getline(myfile, buffer);
		//if(buffer.find("()"))  {
		//	cout << buffer << endl;
		//}
		if (buffer.find("END") ){//&& buffer.find("tensorflow")) { 
			cout << "\e[92m" << buffer << endl;
		}
		else if (buffer.find("0.000")) {
			cout << "\e[91m" << buffer << endl;
		}
		else cout << "\e[91m" << buffer << endl; 
	}
}
