package timetracker

import (
	"time"
	"fmt"
)

func TimeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	fmt.Printf("%s заняло %s\n", name, elapsed)
}