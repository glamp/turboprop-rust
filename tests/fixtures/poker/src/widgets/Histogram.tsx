import { Box, Stack, Typography } from "@mui/material";
import { countBy, round } from "lodash";
import { Player } from "./Player";
import { PlayingCard } from "./PlayingCard";

export const Histogram: React.FC<{ players: Player[] }> = ({ players }) => {
  const estimates = players
    .map((player) => player.estimate)
    .filter(Boolean) as number[];

  const hist = countBy(estimates);
  const average = estimates.reduce((acc, x) => acc + x, 0) / estimates.length;

  /**
  const estimateStdev = Math.sqrt(
    estimates.reduce(
      (acc, x) =>
        acc +
        (x - estimates.reduce((acc, x) => acc + x, 0) / estimates.length) ** 2,
      0
    ) / estimates.length
  );
  const estimateQuality = 1 - estimateStdev / 80;
  let estimateQualityLabel = "Meh";
  let progressColor = "error";
  if (estimateQuality > 0.8) {
    estimateQualityLabel = "Great";
    progressColor = "success";
  } else if (estimateQuality > 0.6) {
    estimateQualityLabel = "Good";
    progressColor = "secondary";
  } else if (estimateQuality > 0.4) {
    estimateQualityLabel = "Ok";
    progressColor = "warning";
  } else {
    estimateQualityLabel = "Meh";
    progressColor = "error";
  }
   */

  return (
    <Stack
      direction="row"
      spacing={8}
      alignItems="center"
      justifyContent="center"
    >
      <Stack direction="row" alignItems="center" spacing={8}>
        <Stack direction="row" spacing={2}>
          {Object.keys(hist)
            .sort((a, b) => Number(a) - Number(b))
            .map((key) => {
              return (
                <Stack
                  key={`histogram-${key}`}
                  direction="column"
                  spacing={1}
                  alignItems="center"
                  justifyContent="flex-end"
                >
                  <Typography>{hist[key]}</Typography>
                  <Box
                    borderRadius={1}
                    bgcolor={
                      hist[key] === Math.max(...Object.values(hist))
                        ? "secondary.main"
                        : "primary.main"
                    }
                    height={hist[key] * 30}
                    width={20}
                  />
                  <PlayingCard isFlipped={false} text={key} />
                </Stack>
              );
            })}
        </Stack>
        <Stack
          direction="column"
          spacing={1}
          justifyContent="center"
          alignItems="center"
          sx={{
            borderWidth: 4,
            borderColor: "primary.main",
            borderStyle: "solid",
            p: 3,
            borderRadius: "100%",
          }}
        >
          <Typography variant="h6">Average</Typography>
          <Typography variant="h4">{round(average, 1)}</Typography>
        </Stack>
      </Stack>
      {/* <Stack direction="column" spacing={1} alignItems="center">
        <Typography variant="body1">Alignment</Typography>
        <Box sx={{ position: "relative", display: "inline-flex" }}>
          <CircularProgress
            variant="determinate"
            color={progressColor}
            size={100}
            value={100 * estimateQuality}
          />
          <Box
            sx={{
              top: 40,
              left: 0,
              right: 0,
              position: "absolute",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Typography
              variant="body2"
              component="div"
              color="text.primary"
              fontWeight="bold"
            >
              {estimateQualityLabel}
            </Typography>
          </Box>
        </Box>
      </Stack> */}
    </Stack>
  );
};

export default Histogram;
