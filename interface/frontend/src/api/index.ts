import { backendApi } from './backend'
import { sampleApi } from './sample'
import type { SearchApi } from './types'

export { SAMPLE_QUERY } from './sample'
export type * from './types'

export function getSearchApi(useSampleData: boolean): SearchApi {
  return useSampleData ? sampleApi : backendApi
}
