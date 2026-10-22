import { siteConfig } from "../config";

// en-CA formats as YYYY-MM-DD.
const seoulDate = new Intl.DateTimeFormat("en-CA", {
  timeZone: siteConfig.timezone,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

/** Formats a date as `YYYY-MM-DD` in the site timezone (Asia/Seoul). */
export function formatSeoulDate(date: Date): string {
  return seoulDate.format(date);
}
