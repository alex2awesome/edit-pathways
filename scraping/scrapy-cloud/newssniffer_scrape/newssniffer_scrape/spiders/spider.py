import scrapy
from scrapy_selenium import SeleniumRequest
import pandas as pd
import glob, os
from tqdm.auto import tqdm
from w3lib.html import remove_tags
import re

here = os.path.dirname(os.path.realpath(__file__))


class BaseSpider(scrapy.Spider):
    def __init__(self, num_splits=1, split_num=1, **kwargs):
        super().__init__(**kwargs)  # python3
        num_splits = os.environ.get('NUM_SPLITS') or num_splits
        split_num = os.environ.get('SPLIT_NUM') or split_num
        self.num_splits = num_splits if isinstance(num_splits, int) else int(num_splits)
        self.split_num = split_num if isinstance(split_num, int) else int(split_num)

    def get_scrapy_cache_dir(self):
        local_cache_dir = os.path.join(here, '../..', 'output_dir')
        if os.path.exists(local_cache_dir):
            return local_cache_dir

        cache_dir_on_docker = '/app/output_dir/'
        if os.path.exists(cache_dir_on_docker):
            return cache_dir_on_docker

        self.logger.error('paths %s or %s don\'t exist.' % (local_cache_dir, cache_dir_on_docker))

    def split_df(self, df):
        if (self.num_splits == 1) and (self.split_num == 1):
            return df

        ## distribute
        splits = len(df) / self.num_splits
        start = int(splits * self.split_num)
        end = int(splits * (self.split_num + 1))
        self.logger.info('split num: %s/%s, collecting range %s-%s' % (
            self.split_num, self.num_splits, start, end
        ))

        return df.iloc[start:end]

    def start_requests(self):
        urls = self.get_urls()
        self.pbar = tqdm(total=len(urls))
        for url in urls:
            yield SeleniumRequest(
                url=url,
                wait_time=3,
                callback=self.parse,
            )

    def extract_table_from_response(self, response):
        table_output = []
        for row in response.css('tr'):
            row_ouput = []
            for cell in row.css('td'):
                if len(cell.css('a')) > 0:
                    cell_link = cell.xpath('a/@href').get()
                    row_ouput.append(cell_link)
                cell_value = remove_tags(cell.get()).strip()
                row_ouput.append(cell_value)
            table_output.append(row_ouput)
        return pd.DataFrame(list(filter(lambda x: len(x) != 0, table_output)))


class NewsSnifferSpider(BaseSpider):
    name = 'newssniff-article-scraper'
    custom_settings = {
        'FEED_FORMAT': 'jsonlines',
        # 'FEED_URI': "working_dir/%(batch_id)d-articles-%(batch_time)s.json"
    }

    def get_files_to_get(self):
        """Get the ids and the diffs that we need to get."""
        cache_dir = self.get_scrapy_cache_dir()
        search_pages = glob.glob(os.path.join(cache_dir, 'newssniffer-index-*'))
        files = list(map(lambda x: pd.read_csv(x, index_col=0), search_pages))
        files_df = pd.concat(files).reset_index(drop=True)
        files_df['file_id'] = files_df['newssniff_link'].str.split('/').str.get(4).astype(int)

        ## get max version per file
        max_diff_per_file_id = (
            files_df[['version', 'file_id']]
                .assign(version=lambda df: df['version'].astype(int))
                .groupby('file_id')
                ['version'].max())
        ## merge
        files_df = (files_df.merge(
                    right=max_diff_per_file_id.astype(int).to_frame('num_diffs_per_file'),
                    left_on='file_id', right_index=True
                ).drop_duplicates('file_id'))

        return self.split_df(files_df)

    def get_existing_ids(self):
        """Get the ids for the files we've already gotten."""
        ## path for the requests/bs4 output
        cache_dir = self.get_scrapy_cache_dir()
        ## check if cache already exists... (short-circuit creating-from-scrath)
        file_id_csv = os.path.join(cache_dir, 'file-ids-scraped.csv')
        if os.path.exists(file_id_csv):
            return pd.read_csv(file_id_csv, index_col=0, squeeze=True)

        a_vers_fs = glob.glob(os.path.join(cache_dir, 'article-versions-*'))
        a_vers_dfs = list(map(lambda x: pd.read_csv(x, usecols=['file_id'], squeeze=True), a_vers_fs))

        ## another path for the scrapy output
        scrapy_cache_dir = self.get_scrapy_cache_dir()
        a_vers_fs = (glob.glob(os.path.join(scrapy_cache_dir, '*-articles-*')) +
                             glob.glob(os.path.join(scrapy_cache_dir, '*filename*')))
        for f in a_vers_fs:
            try:
                df = pd.read_json(path_or_buf=f, orient='records', lines=True)
            except:
                df = pd.read_json(path_or_buf=f, orient='records', lines=False)
            if len(df)>0:
                a_vers_dfs.append(df['article_id'])

        existing_files = pd.concat(a_vers_dfs)
        return existing_files.drop_duplicates().astype(int)

    def filter_files(self, to_get_df, existing_ids, min_p=.01, max_p=.99):
        """ Filter files to just include those with diffs in the min & max percentiles, and
        those we haven't already gotten.
        """
        min_cut, max_cut = to_get_df['num_diffs_per_file'].loc[lambda s: s != 0].quantile([min_p, max_p])
        to_scrape_table = (
            to_get_df
                 .loc[lambda df: df['num_diffs_per_file'].pipe(lambda s: (s >= min_cut) & (s <= max_cut))]  ## 1, 30
                 [['file_id', 'num_diffs_per_file']]
                 .drop_duplicates()
                .loc[lambda df: ~df['file_id'].astype(int).isin(existing_ids.values)]
        )
        return to_scrape_table

    def get_urls(self):
        self.logger.info('Getting search ids...')
        files_to_get = self.get_files_to_get() ## get files to parse
        self.logger.info('Num file-ids to fetch: %s' % len(files_to_get))
        self.logger.info('Getting existing files...')
        existing_ids = self.get_existing_ids() ## get already parsed files
        self.logger.info('Filtering out %s existing ids...' % len(existing_ids))
        to_scrape_table = self.filter_files(to_get_df=files_to_get, existing_ids=existing_ids)
        self.logger.info('%s ids remaining to scrape...' % len(to_scrape_table))

        ## generate urls
        urls_to_scrape = []
        for file_id, num_diffs in to_scrape_table.itertuples(index=False):
            v = list(range(num_diffs + 1))
            for s, e in list(zip(v[:-1], v[1:])):
                url = 'https://www.newssniffer.co.uk/articles/%s/diff/%s/%s' % (file_id, s, e)
                urls_to_scrape.append(url)
        return urls_to_scrape

    def parse(self, response):
        source = response.selector.css('cite::text').get()
        request_url = response.url
        article_id = response.url.split('/')[4]
        article_url = response.selector.css('cite a::text').get()

        ## get the rest of the article
        version_tables = pd.read_html(response.body.decode('utf-8'))[0]
        version_table_flat = (version_tables
              .apply(lambda s: ('</p><p>').join(s.dropna().tolist()))
              .reset_index()
              .rename(columns={'level_0': 'version', 'level_1': 'title', 'level_2': 'time', 0: 'text'})
        )

        ## merge
        version_table_flat['article_url'] = article_url
        version_table_flat['request_url'] = request_url
        version_table_flat['source'] = source
        version_table_flat['article_id'] = article_id
        output = version_table_flat.to_dict(orient='records')
        self.pbar.update(1)
        for o in output:
            yield o


class NewsSnifferSearchSpider(BaseSpider):
    name = 'newssniff-search-scraper'

    def get_urls(self):
        start_page = 200
        last_page = 239295
        ## get all pages
        all_search_pages = pd.Series(list(range(start_page, last_page)))
        self.logger.info('%s search pages total...' % len(all_search_pages))

        ## get pages already searched
        cache_dir = self.get_scrapy_cache_dir()
        search_file_csv = os.path.join(cache_dir, 'search-pages-scraped.csv')
        searched_files = glob.glob(os.path.join(cache_dir, 'newss*'))
        if os.path.exists(search_file_csv):
            searched_pages = pd.read_csv(search_file_csv, index_col=0, squeeze=True)
        elif len(searched_files) > 0:
            search_pages = list(map(lambda x: pd.read_csv(x, usecols=['search_page'], squeeze=True), searched_files))
            searched_pages = pd.concat(search_pages).drop_duplicates().values
        self.logger.info('Already scraped %s pages...' % len(searched_pages))

        ## filter out the pages already scraped and get a final list of pages
        to_search = all_search_pages.loc[lambda s: ~s.isin(searched_pages)]
        to_search = self.split_df(to_search)
        self.logger.info("%s pages to search..." % len(to_search))

        ## form URLs
        url_base = 'https://www.newssniffer.co.uk/versions?page=%s'
        return list(map(lambda x: url_base % x, to_search))

    def parse(self, response):
        page_df = self.extract_table_from_response(response)
        page_df.columns = ['newssniff_link', 'headline', 'version', 'outlet', 'last_updated']
        page = int(re.search('page=(\d+)', response.url)[1])
        page_df['search_page'] = page
        output = page_df.to_dict(orient='records')
        self.pbar.update(1)
        for o in output:
            yield o


class NewsSnifferAritlcePageSpider(BaseSpider):
    name = 'newssniff-article-page'

    def get_existing_ids(self):
        """Get the ids for the files we've already gotten."""
        ## path for the requests/bs4 output
        cache_dir = self.get_scrapy_cache_dir()
        ## check if cache already exists... (short-circuit creating-from-scrath)
        file_id_csv = os.path.join(cache_dir, 'file-ids-scraped.csv')
        if os.path.exists(file_id_csv):
            return pd.read_csv(file_id_csv, index_col=0, squeeze=True)

    def get_urls(self, final_pass=True):
        if not final_pass:
            article_id_range = pd.Series(list(range(35, 2058846)))
            existing_ids = self.get_existing_ids()
            to_get = article_id_range.loc[lambda s: ~s.isin(existing_ids)]
        else:
            cache_dir = self.get_scrapy_cache_dir()
            fn = os.path.join(cache_dir, '2021-01-17__article-page-ids-to-get.csv')
            to_get = pd.read_csv(fn, index_col=0, squeeze=True)
        to_get = self.split_df(to_get)
        base_url = 'https://www.newssniffer.co.uk/articles/%s'
        return list(map(lambda x: base_url % x, to_get.values))

    def parse(self, response, **kwargs):
        article_id = response.url.split('/')[-1]
        article_df = self.extract_table_from_response(response)

        if len(article_df.columns) > 0:
            article_df.columns = ['version_url', 'version', 'time_created', 'time_delta', 'title']
            source = response.selector.css('cite::text').get()
            article_url = response.selector.css('cite a::text').get()
            ##
            article_df['article_url'] = article_url
            article_df['article_id'] = article_id
            article_df['source'] = source
            ##
            output = article_df.to_dict(orient='records')
            for o in output:
                yield o
        else:
            yield {
                'article_id': article_id,
                'html': response.text,
                'parse_error': True,
            }
        self.pbar.update(1)

class NewsSnifferVersionPageSpider(BaseSpider):
    name = 'newssniffer-version-page'

    def get_urls(self):
        cache_dir = self.get_scrapy_cache_dir()
        url_file_path = os.path.join(cache_dir, 'version-urls.csv')
        if os.path.exists(url_file_path):
            url_df = pd.read_csv(url_file_path, index_col=0, squeeze=True)
        url_df = self.split_df(url_df)
        return url_df.tolist()

    def parse(self, response, **kwargs):
        url = response.url
        ps = response.css('p')
        html = list(map(lambda x: x.get(), ps))
        html = ''.join(html)
        self.pbar.update(1)
        yield {
            'url': url,
            'html': html
        }