import { Histogram } from "@/widgets/Histogram";
import { InputName } from "@/widgets/InputName";
import { Player, PlayerPresenter } from "@/widgets/Player";
import { SelectEstimate } from "@/widgets/SelectEstimate";
import {
  Button,
  Container,
  Dialog,
  DialogContent,
  Stack,
  Typography,
  Switch,
  FormControlLabel
} from "@mui/material";
import { useChannel, usePresence } from "ably/react";
import { useRouter } from "next/router";
import * as React from "react";
import { useAsync, useDebounce } from "react-use";
import { Fetcher } from "swr";

const fetcher: Fetcher<Round, string> = async (url) => {
  return (await (await fetch(url)).json()) as Round;
};

export type Round = {
  estimate: number;
  players: Player[];
};

const Room: React.FC<{
  player: Player;
  setPlayer: (player: Player) => void;
}> = ({ player, setPlayer }) => {
  const router = useRouter();
  const roomId = router.query.id as string;
  const [estimate, setEstimate] = React.useState<number | undefined>();
  const [count, setCount] = React.useState(3);
  const [isActive, setIsActive] = React.useState(false);
  const [reveal, setReveal] = React.useState(false);

  const channelName = `room/${roomId}`;
  const { channel, ably } = useChannel(channelName);
  const { presenceData, updateStatus } = usePresence<Player>(
    channelName,
    player
  );

  const nonGuests = presenceData.filter((p) => p.data.name !== "Guest");
  const guests = presenceData.filter((p) => p.data.name === "Guest");
  const sequence = presenceData.filter((p) => p.data.isLeader)[0]?.data?.sequence;
  const [fibSequence, setFibSequence] = React.useState(sequence ? sequence : false);
  
  useAsync(async () => {
    if (!player) {
      return;
    }
    await updateStatus(player);
  }, [player]);

  useDebounce(
    async () => {
      if (nonGuests.length === 0) {
        return;
      }
      // if the player is the only player in the room, then make them the leader
      if (
        player &&
        !player.isLeader &&
        nonGuests.map((p) => p.data.name === player.name).length === 1
      ) {
        setPlayer({
          ...player,
          isLeader: true,
        });
      }
    },
    100,
    [player, presenceData]
  );

  useChannel(channelName, "reveal", () => {
    setCount(3);
    setIsActive(true);
  });

  useChannel(channelName, "new-round", () => {
    setReveal(false);
    setPlayer({
      ...player,
      estimate: null,
    });
    setEstimate(undefined);
  });

  useChannel(channelName, "sequence-change", () => {
    setPlayer({
      ...player,
      sequence: fibSequence,
    });
  });

  useAsync(async () => {
    if (!player || !estimate || !roomId) {
      return;
    }
    await updateStatus({
      ...player,
      estimate,
    });
  }, [estimate]);

  const onClickReveal = React.useCallback(async () => {
    ably.channels.get(channelName).publish("reveal", { ts: new Date() });
  }, [ably, channelName]);

  const onClickNewRound = React.useCallback(async () => {
    ably.channels.get(channelName).publish("new-round", { ts: new Date() });
  }, [ably, channelName]);

  const onClickSequenceToggle = React.useCallback(async () => {
    ably.channels.get(channelName).publish("sequence-change", { ts: new Date() });
  }, [ably, channelName]);

  React.useEffect(() => {
    let timerId: NodeJS.Timeout | undefined;

    if (isActive && count > 0) {
      timerId = setTimeout(() => {
        setCount(count - 1);
      }, 1000);
    } else if (count === 0) {
      setIsActive(false); // Reset to initial state when countdown finishes
      setReveal(true);
    }

    return () => {
      if (timerId) clearTimeout(timerId);
    };
  }, [count, isActive, fibSequence]);

  const players = nonGuests.map((msg) => msg.data as Player);

  if (!presenceData.length) {
    // center vertically and horizontally using stack
    return (
      <Container maxWidth="lg">
        <Stack
          direction="column"
          justifyContent="center"
          alignItems="center"
          sx={{ height: "80vh" }}
        >
          <Typography variant="h4">Joining room...</Typography>
        </Stack>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Stack
        spacing={{ xs: 1, sm: 2 }}
        direction="row"
        justifyContent="center"
        useFlexGap
        flexWrap="wrap"
        p={2}
        sx={{ width: "100%" }}
      >
        <FormControlLabel
          control={
            <Switch
              checked={fibSequence}
              onChange={() => {
                setFibSequence(!fibSequence);
                onClickSequenceToggle();
              }}
              disabled={reveal || !player?.isLeader}
            />
          }
          label="Fibonacci"
        />
      </Stack>
      <Stack direction="column" spacing={2}>
        <Stack
          spacing={{ xs: 1, sm: 2 }}
          direction="row"
          justifyContent="center"
          useFlexGap
          flexWrap="wrap"
          p={2}
          sx={{ width: "100%" }}
        >
          {players
            .filter((x) => x)
            .map((p, index) => (
              <PlayerPresenter
                key={`player-${index}`}
                {...p}
                isMe={p.name === player?.name}
                reveal={!reveal}
              />
            ))}
        </Stack>

        {!reveal && player.name !== "Guest" && (
          <SelectEstimate estimate={estimate} setEstimate={setEstimate} sequence={sequence} />
        )}

        <Dialog open={isActive}>
          <DialogContent sx={{ p: 4 }}>
            <Typography variant="h4">
              {count > 0 ? count : "Reveal!"}
            </Typography>
          </DialogContent>
        </Dialog>

        {player?.isLeader && !reveal && (
          <Stack direction="column" alignItems="center">
            <Button
              size="large"
              onClick={onClickReveal}
              disabled={isActive}
              variant="contained"
            >
              Reveal
            </Button>
          </Stack>
        )}

        {reveal && <Histogram players={players} />}
        {player?.isLeader && reveal && (
          <Stack direction="column" alignItems="center">
            <Button
              size="large"
              onClick={onClickNewRound}
              disabled={isActive}
              variant="contained"
            >
              New Round
            </Button>
          </Stack>
        )}
      </Stack>
      {/* <pre>{JSON.stringify(players, null, 2)}</pre> */}
    </Container>
  );
};

export const RoomWithLogin: React.FC = () => {
  const [player, setPlayer] = React.useState<Player>();
  if (!player) {
    return (
      <Container maxWidth="lg">
        <InputName onComplete={(newPlayer) => setPlayer(newPlayer)} />
      </Container>
    );
  }

  return <Room player={player} setPlayer={setPlayer} />;
};

export default RoomWithLogin;
