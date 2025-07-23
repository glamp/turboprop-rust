import { PlayingCard } from "./PlayingCard";
import { Stack } from "@mui/material";
import * as React from "react";

export const SelectEstimate: React.FC<{
  estimate: number | undefined;
  setEstimate: (estimate: number) => void;
  sequence: boolean | undefined;
}> = ({ estimate, setEstimate, sequence }) => {
  let sequenceArray = [10, 20, 40, 60, 80, 120, 180, 240, 480];
  if (sequence) {
    sequenceArray = [1, 3, 5, 8, 13, 21];
  }
  
  return (
    <Stack
      direction="row"
      spacing={2}
      justifyContent="center"
      useFlexGap
      flexWrap="wrap"
      sx={{
        p: { xs: 2, sm: 0 },
      }}
    >
    {
      sequenceArray.map(function(data) {
        return (
          <PlayingCard
            onClick={() => setEstimate(data)}
            selected={estimate === data}
            selectable
            isFlipped={false}
            text={data}
            key={data}
          />
        )
      })
    }
    </Stack>
  );
};

export default SelectEstimate;
