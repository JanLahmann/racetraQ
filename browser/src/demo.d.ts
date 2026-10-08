/** Types for the server demo's canvas renderer (racetraq/web/js/race.js),
 *  which the browser edition imports unchanged via the @demo alias. */
declare module '@demo/race.js' {
  export interface RenderCar {
    id: string;
    kind: string;
    label?: string;
    x: number;
    y: number;
    theta: number;
    v: number;
    rays?: number[];
    ghost?: boolean;
    off_track?: boolean;
  }
  export interface TrackPayload {
    name: string;
    half_width: number;
    total_length: number;
    checkpoints: number[];
    theme: { surface?: string; edge?: string };
    start: { x: number; y: number; theta: number };
    centerline: [number, number][];
    left: [number, number][];
    right: [number, number][];
  }
  export const KIND_COLORS: Record<string, string>;
  export const STAGE_COLORS: string[];
  export type CameraView = 'top' | 'chase' | 'cockpit';
  export const CAMERA_VIEWS: CameraView[];
  export class RaceRenderer {
    constructor(canvas: HTMLCanvasElement);
    running: boolean;
    showRays: boolean;
    camera: CameraView;
    minimapCorner: string;
    setCamera(view: CameraView): void;
    cycleCamera(): CameraView;
    setTrack(payload: TrackPayload): void;
    pushState(msg: { cars: RenderCar[] }): void;
    setMode(mode: string): void;
    stageColor(label: string): string;
    addEffect(kind: string, carId: string): void;
    start(): void;
  }
}
