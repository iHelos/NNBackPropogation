package data_helper

import (
	"encoding/csv"
	"bufio"
	"io"
	"fmt"
	"os"
	"strconv"
)

const MAX_DARKNESS float64 = 255

func GetTrainData(path string, header bool) (train_x [][]float64, train_y [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		value, err := strconv.Atoi(record[0])
		value_arr := make([]float64, 10)
		value_arr[value] = 1.
		if err != nil {
			fmt.Println("Not valid csv file")
			os.Exit(1)
		}
		image := record[1:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		train_y = append(train_y, value_arr)
		train_x = append(train_x, image_normalized)
	}
	return
}

func GetTestData(path string, header bool) (test_x [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		image := record[0:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		test_x = append(test_x, image_normalized)
	}
	return
}

func GetLabel(arr []float64) (label int) {
	for i, value := range arr{
		if value == 1{
			label = i
			return
		}
	}
	return
}
