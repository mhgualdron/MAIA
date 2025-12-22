package main

import (
	"fmt"
	"reflect"
)

func main(){
	// Comentario de primer tipo
	/*
	Comentario de segundo tipo
	*/
	/*
	fmt.Println("Hello World")
	fmt.Println("Bye World")
	*/

	fmt.Println("Hello World")

	// Variables

	var myString string = "Esto es una cadena de texto"
	fmt.Println(myString)

	var myInt int8
	myInt = 11
	//myInt = myInt + 9
	fmt.Println(myInt)


	fmt.Println(myString, myInt)

	fmt.Println(reflect.TypeOf(myInt))

	var myFloat float32
	myFloat=60.5
	fmt.Println(myFloat)
	fmt.Println(reflect.TypeOf(myFloat))


	const myConstante = "Constante1"

	fmt.Println(myConstante)


		// IF ELSE
	
	if myInt == 10 && myConstante == "Constante1"{
		fmt.Println("Se cumplen ambas")
	} else if myInt == 11 || myConstante == "Constante1"{
		fmt.Println("Se cumple una de las dos")
	} else {
		fmt.Println("El valor no es 10")
	}


	// Arrays

	var myArray [4]int = [4]int{1,2,3,4}
	//myArray[0] = 1

	myArray[3] = 999

	fmt.Println(myArray)



	myMap := make(map[string]int) //Clave string y valor tipo Int

	myMap["Colombia"] = 777
	myMap["Españita"] = 111
	myMap["Error"] = 9.0 //No es un error, 9.0 es 9 para Go

	fmt.Println(myMap)

	fmt.Println("Este es el mapa de Colombia", myMap["Colombia"])



	myMap2 := map[string]int{
		"Colombia":77,
		"Españita":11}
	fmt.Println(myMap2)


	for index:= 1; index < 5; index++{
		if index == 3{
			continue
		}
		fmt.Println("Este es el index",index)
	}


	for index, value := range myArray{
		fmt.Println("Este es el index",index,"este el valor", value)
	}

	myFunction()
	fmt.Println(retornar())



	type myStructure struct {		name string
		age int
		country string
	}
	myStruct := myStructure{"Mateo Hernandez", 23, "Colombia"}
	fmt.Println(myStruct)

}

func myFunction(){
	fmt.Println("Esto es una función")
}

func retornar()string{
	return "Retorna este string"
}